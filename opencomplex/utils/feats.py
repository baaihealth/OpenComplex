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

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union
from einops import rearrange

from opencomplex.np import protein
import opencomplex.np.residue_constants as rc
from opencomplex.utils.complex_utils import ComplexType
from opencomplex.utils.rigid_utils import Rotation, Rigid
from opencomplex.utils.tensor_utils import (
    batched_gather,
    one_hot,
    tree_map,
    tensor_tree_map,
)


def dense_atom_to_all_atom(dense_atom, batch):
    all_atom_data = batched_gather(
        dense_atom,
        batch["residx_all_to_dense"],
        dim=-2,
        no_batch_dims=len(dense_atom.shape[:-2]),
    )

    all_atom_data = all_atom_data * batch["all_atom_exists"][..., None]

    return all_atom_data


def build_template_angle_feat(template_feats, c_butype=rc.restype_num):
    template_butype = template_feats["template_butype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    if "template_alt_torsion_angles_sin_cos" in template_feats:
        alt_torsion_angles_sin_cos = template_feats[
            "template_alt_torsion_angles_sin_cos"
        ]
        template_angle_feat = torch.cat(
            [
                nn.functional.one_hot(template_butype, c_butype + 2),
                rearrange(torsion_angles_sin_cos, '... a b -> ... (a b)', b = 2),
                rearrange(alt_torsion_angles_sin_cos, '... a b -> ... (a b)', b = 2),
                torsion_angles_mask,
            ],
            dim=-1,
        )
    else:
        template_angle_feat = torch.cat(
            [
                nn.functional.one_hot(template_butype, c_butype + 2),
                torsion_angles_sin_cos[..., 0],
                torsion_angles_sin_cos[..., 1],
                torsion_angles_mask,
            ],
            dim=-1,
        )

    return template_angle_feat


def build_template_pair_feat(
    batch, 
    min_bin, max_bin, no_bins, 
    multichain_mask=None,
    use_unit_vector=False, 
    c_butype=rc.restype_num,
    eps=1e-20, inf=1e8
):
    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    if multichain_mask is not None:
        template_mask_2d *= multichain_mask

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = batch["template_pseudo_beta"]
    dgram = torch.sum(
        (tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    to_concat = [dgram, template_mask_2d[..., None]]

    butype_one_hot = nn.functional.one_hot(
        batch["template_butype"],
        c_butype + 2,
    )

    n_res = batch["template_butype"].shape[-1]
    to_concat.append(
        butype_one_hot[..., None, :, :].expand(
            *butype_one_hot.shape[:-2], n_res, -1, -1
        )
    )
    to_concat.append(
        butype_one_hot[..., None, :].expand(
            *butype_one_hot.shape[:-2], -1, n_res, -1
        )
    )

    n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))

    t_aa_masks = batch["template_all_atom_mask"]
    template_mask = (
        t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    )
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    if multichain_mask is not None:
        template_mask_2d *= multichain_mask

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    
    if(not use_unit_vector):
        unit_vector = unit_vector * 0.
    
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])

    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_extra_msa_feat(batch, complex_type=ComplexType.PROTEIN):
    num_class = 27 if complex_type == ComplexType.MIX else 23
    msa_1hot = nn.functional.one_hot(batch["extra_msa"], num_class)
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    butype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[butype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    butype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[butype, ...]

    # [*, N, 14]
    group_mask = group_idx[butype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[butype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[butype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
