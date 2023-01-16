# Copyright 2022 BAAI
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, Optional, Union

import ml_collections
import torch
from torch import nn

from einops import repeat

from opencomplex.config.references import NUM_FULL_RES, NUM_RES
from opencomplex.loss import (
    loss_fns_rna,
    loss_fns_protein,
)
from opencomplex.loss.loss_utils import (
    softmax_cross_entropy,
    sigmoid_cross_entropy,
    lddt,
)
from opencomplex.utils.rigid_utils import Rigid, Rotation
from opencomplex.loss.permutation import multichain_permutation_alignment
from opencomplex.utils.complex_utils import split_protein_rna_pos
import copy

def split_data(batch, out, pos, feat_schema):
    ret = {}
    if len(pos) == 0:
        return ret

    ret["pred_dense_positions"] = out["sm"]["positions"][..., pos, :, :]
    ret["distogram_logits"] = out["distogram_logits"][..., pos, pos, :]
    ret["angles_sin_cos"] = out["sm"]["angles"][..., pos, :, :]
    ret["unnormalized_angles_sin_cos"] = out["sm"]["unnormalized_angles"][..., pos, :, :]

    for k, v in batch.items():
        if k not in feat_schema:
            continue
        ret[k] = v
        for i, dim_size in enumerate(feat_schema[k]):
            offset = ret[k].ndim - len(feat_schema[k])
            if dim_size == NUM_RES or dim_size == NUM_FULL_RES:
                ret[k] = torch.index_select(ret[k], offset + i, pos)

    if "asym_id" not in batch:
        ret["asym_id"] = None

    for key in ['dis_n', 'dis_c4', 'dis_p', 'omg', 'theta']:
        if "geometry_head" in out:
            ret["pred_" + key] = out["geometry_head"]["pred_" + key][..., pos, :, :][..., pos, :]

    if "torsion_head" in out:
        ret["angles_dis"] = out["torsion_head"]["angles_dis"][..., pos, :, :]
    
    return ret


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2
    
    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def experimentally_resolved_loss(
    logits: torch.Tensor,
    all_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    errors = sigmoid_cross_entropy(logits, all_atom_mask)
    loss = torch.sum(errors * all_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(all_atom_exists, dim=(-1, -2))).view(loss.size(0),1)
    loss = torch.sum(loss, dim=-1)
    
    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    loss = torch.mean(loss)
 
    return loss


def masked_msa_loss(logits, true_msa, bert_mask, eps=1e-8, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, N_seq, N_res, 7] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask
    Returns:
        Masked MSA loss
    """
    N1, N2 = logits.shape[1:3]
    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_msa, num_classes=int(logits.shape[-1]))
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    # numerical stability
    # NOTE: is this necessary?
    denom = torch.max(denom, torch.Tensor([N1*N2/20]).to(denom.device))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss

def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    # NOTE: ca_pos
    ca_pos = 1
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def tm_loss(
    logits,
    final_affine_tensor,
    backbone_rigid_tensor,
    backbone_rigid_mask,
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    pred_affine = final_affine_tensor
    backbone_affine = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_affine)) ** 2, dim=-1
    )

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    square_mask = (
        backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the loss dimension
    loss = torch.mean(loss)

    return loss

    
def compute_renamed_ground_truth(
    dense_pred_positions: torch.Tensor,
    dense_atom_gt_positions: torch.Tensor,
    dense_atom_gt_exists: torch.Tensor,
    dense_atom_alt_gt_positions: torch.Tensor = None,
    dense_atom_is_ambiguous: torch.Tensor = None,
    dense_atom_alt_gt_exists: torch.Tensor = None,
    eps=1e-10,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    if dense_atom_alt_gt_positions is None:
        return {
            "alt_naming_is_better": None,
            "renamed_dense_atom_gt_positions": dense_atom_gt_positions,
            "renamed_dense_atom_gt_exists": dense_atom_gt_exists,
        }


    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_pred_positions[..., None, :, None, :]
                - dense_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_atom_gt_positions[..., None, :, None, :]
                - dense_atom_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_atom_alt_gt_positions[..., None, :, None, :]
                - dense_atom_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    mask = (
        dense_atom_gt_exists[..., None, :, None]
        * dense_atom_is_ambiguous[..., None, :, None]
        * dense_atom_gt_exists[..., None, :, None, :]
        * (1.0 - dense_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = dense_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * dense_atom_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * dense_atom_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * dense_atom_gt_exists + alt_naming_is_better[..., None] * dense_atom_alt_gt_exists

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_dense_atom_gt_positions": renamed_atom14_gt_positions,
        "renamed_dense_atom_gt_exists": renamed_atom14_gt_mask
    }



def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    pair_mask: torch.Tensor = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            pair_mask:
                A (num_frames, num_positions) mask to use in the loss, useful
                for separating intra from inter chain losses.
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]
    if pair_mask is not None:
        normed_error *= pair_mask

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)

    if pair_mask is not None:
        normed_error = torch.sum(normed_error, dim=(-2, -1))
        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        normalization_factor = torch.sum(mask, dim=(-2, -1))
        normed_error = normed_error / (eps + normalization_factor)
    else:
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error

def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    pred_aff: Rigid,
    enable_use_clamped_fape: bool = False,
    use_clamped_fape: Optional[torch.Tensor] = None,
    pair_mask:torch.Tensor = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    # import pdb; pdb.set_trace()
    # DEBUG
    # if len(pred_aff.shape) == len(gt_aff.shape):
    #     pred_aff = Rigid.stack([pred_aff, 
    #                             Rigid(
    #                                 rots=Rotation(
    #                                     rot_mats=torch.zeros_like(pred_aff.get_rots().get_rot_mats())
    #                                     ),
    #                                 trans=torch.zeros_like(pred_aff.get_trans())
    #                             )], dim=-1)

    if backbone_rigid_mask.shape[-1] > 1:
        # concat in N dim, from [*, N, 2] to [*, N*2]
        gt_aff = Rigid.cat([gt_aff[..., 0], gt_aff[..., 1]], dim=-1)
        pred_aff = Rigid.cat([pred_aff[..., 0], pred_aff[..., 1]], dim=-1)
        backbone_rigid_mask = torch.cat([backbone_rigid_mask[..., 0], backbone_rigid_mask[..., 1]], dim=-1)

        if pair_mask is not None:
            pair_mask = repeat(pair_mask, '... h w -> ... (a h) (b w)', a=2, b=2)
    else:
        backbone_rigid_mask = backbone_rigid_mask[..., 0]
        gt_aff = gt_aff[..., 0]


    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        pair_mask=pair_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    if enable_use_clamped_fape and use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (1 - use_clamped_fape)

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def sidechain_loss(
    sidechain_frames: torch.Tensor, # predicted frames
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor, # ground truth frames
    rigidgroups_gt_exists: torch.Tensor,
    renamed_dense_atom_gt_positions: torch.Tensor,
    renamed_dense_atom_gt_exists: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor=None,
    alt_naming_is_better: torch.Tensor=None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    sidechain_frames = sidechain_frames[-1]

    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)

    if alt_naming_is_better is not None:
        renamed_gt_frames = (
            1.0 - alt_naming_is_better[..., None, None, None]
        ) * rigidgroups_gt_frames + alt_naming_is_better[
            ..., None, None, None
        ] * rigidgroups_alt_gt_frames
    else:
        renamed_gt_frames = rigidgroups_gt_frames

    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)

    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)

    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_dense_atom_gt_positions = renamed_dense_atom_gt_positions.view(*batch_dims, -1, 3)
    renamed_dense_atom_gt_exists = renamed_dense_atom_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        pred_frames=sidechain_frames,
        target_frames=renamed_gt_frames,
        frames_mask=rigidgroups_gt_exists,
        pred_positions=sidechain_atom_pos,
        target_positions=renamed_dense_atom_gt_positions,
        positions_mask=renamed_dense_atom_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
    asym_id: torch.Tensor = None,
) -> torch.Tensor:
    if asym_id is not None:
        intra_chain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).float()
        intra_chain_bb_loss = backbone_loss(
            pred_aff=out['sm']['frames'],
            pair_mask=intra_chain_mask,
            **{**batch, **config.intra_chain_backbone},
        )
        interface_bb_loss = backbone_loss(
            pred_aff=out['sm']['frames'],
            pair_mask=1.0 - intra_chain_mask,
            **{**batch, **config.interface_backbone},
        )
        bb_loss = intra_chain_bb_loss + interface_bb_loss
    else:
        bb_loss = backbone_loss(
            pred_aff=out["sm"]["frames"],
            **{**batch, **config.backbone},
        )

    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **compute_renamed_ground_truth(
            dense_pred_positions=out["sm"]["positions"][-1],
            **batch),
        **{**batch, **config.sidechain},
    )

    # DEBUG
    # print("bb loss", bb_loss, "sc loss", sc_loss)
    loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss
    
    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


class OpenComplexLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""
    def __init__(self, config, feat_schema, complex_type):
        super(OpenComplexLoss, self).__init__()
        self.config = config
        self.feat_schema = feat_schema
        self.complex_type = complex_type

    def forward(self, out, batch, _return_breakdown=False):
        # print(out.keys())
        # return torch.sum(out["msa"]), {}
        align_rmsd = None
        if 'asym_id' in batch:
            with torch.no_grad():
                # TODO (yujingcheng): complex permu align, use pseudo beta instead of CA
                batch, align_rmsd = multichain_permutation_alignment(batch, out, self.feat_schema)
                if isinstance(align_rmsd, torch.Tensor):
                    align_rmsd = float(align_rmsd)


        protein_pos, rna_pos = split_protein_rna_pos(batch, self.complex_type)
        protein_data = split_data(batch, out, protein_pos, self.feat_schema)
        rna_data = split_data(batch, out, rna_pos, self.feat_schema)

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**batch, **self.config.experimentally_resolved},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
                asym_id=batch.get('asym_id', None),
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **self.config.plddt_loss,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
            "supervised_chi": lambda: 
                (
                    loss_fns_protein.supervised_chi_loss(
                        **{**protein_data, **self.config.supervised_chi}
                    ) if protein_data else 0.
                ) + \
                (
                    loss_fns_rna.supervised_chi_loss(
                        **{**rna_data, **self.config.supervised_chi}
                    ) if rna_data else 0.
                ),
            "violation": lambda:
                (
                    loss_fns_protein.violation_loss(
                        protein_data,
                        **self.config.violation
                    ) if protein_data else 0.
                ) + \
                (
                    loss_fns_rna.violation_loss(
                        rna_data,
                        **self.config.violation
                    ) if rna_data else 0.
                ),
        }

        if rna_data:
            loss_fns.update(
                {
                    "disgeometry": lambda: loss_fns_rna.geometry_loss(
                        rna_data
                    ),
                    "disbackbone": lambda: loss_fns_rna.disbackbone_loss(
                        rna_data,
                        **self.config.disbackbone,
                    ),
                    "angbackbone": lambda: loss_fns_rna.residue_angle_loss(
                        rna_data,
                        **self.config.angbackbone,
                    ),
                    "distorsion": lambda: loss_fns_rna.torsion_loss(
                        rna_data
                    ),
                }
            )

        if(self.config.tm.enabled):
            loss_fns["tm"] = lambda: tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            # DEBUG
            # print(loss_name, weight)
            loss = loss_fn()
            if(torch.isnan(loss) or torch.isinf(loss)):
                #for k,v in batch.items():
                #    if(torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))):
                #        logging.warning(f"{k}: is nan")
                #logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = torch.tensor(batch["butype"].shape[-1], device=seq_len.device)
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()
        if align_rmsd is not None:
            losses["align_rmsd"] = align_rmsd

        if(not _return_breakdown):
            return cum_loss
        
        return cum_loss, losses