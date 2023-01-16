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

from typing import Dict, Optional, Tuple
from opencomplex.utils.complex_utils import correct_rna_butype

import torch

from opencomplex.np import nucleotide_constants
from opencomplex.utils.tensor_utils import masked_mean
from opencomplex.loss.loss_utils import (
    lddt,
    softmax_cross_entropy
)

def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 27/20, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 27/20)
    residue_index: torch.Tensor,  # (*, N)
    nttype: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # Get the positions of the relevant backbone atoms.
    this_c3_pos = pred_atom_positions[..., :-1, 0, :]
    this_c3_mask = pred_atom_mask[..., :-1, 0]
    this_o3_pos = pred_atom_positions[..., :-1, 6, :]
    this_o3_mask = pred_atom_mask[..., :-1, 6]
    
    next_p_pos = pred_atom_positions[..., 1:, 8, :]
    next_p_mask = pred_atom_mask[..., 1:, 8]
    next_o5_pos = pred_atom_positions[..., 1:, 7, :]
    next_o5_mask = pred_atom_mask[..., 1:, 7]
    
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the o3--p bond.
    o3_p_bond_length = torch.sqrt(
        eps + torch.sum((this_o3_pos - next_p_pos) ** 2, dim=-1)
    )
    
    gt_length = nttype.new_ones(nttype[..., 1:].shape) * nucleotide_constants.between_res_bond_length_o3_p
    gt_stddev = nttype.new_ones(nttype[..., 1:].shape) * nucleotide_constants.between_res_bond_length_stddev_o3_p

    o3_p_bond_length_error = torch.sqrt(eps + (o3_p_bond_length - gt_length) ** 2)
    o3_p_loss_per_residue = torch.nn.functional.relu(
        o3_p_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_o3_mask * next_p_mask * has_no_gap_mask
    o3_p_loss = torch.sum(mask * o3_p_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    o3_p_violation_mask = mask * (
        o3_p_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    c3_o3_bond_length = torch.sqrt(
        eps + torch.sum((this_c3_pos - this_o3_pos) ** 2, dim=-1)
    )
    p_o5_bond_length = torch.sqrt(
        eps + torch.sum((next_p_pos - next_o5_pos) ** 2, dim=-1)
    )

    o3_c3_unit_vec = (this_c3_pos - this_o3_pos) / c3_o3_bond_length[..., None]
    o3_p_unit_vec = (next_p_pos - this_o3_pos) / o3_p_bond_length[..., None]
    p_o5_unit_vec = (next_o5_pos - next_p_pos) / p_o5_bond_length[..., None]

    c3_o3_p_cos_angle = torch.sum(o3_c3_unit_vec * o3_p_unit_vec, dim=-1)
    gt_angle = nucleotide_constants.between_res_cos_angles_c3_o3_p[0]
    gt_stddev = nucleotide_constants.between_res_cos_angles_c3_o3_p[1]
    c3_o3_p_cos_angle_error = torch.sqrt(
        eps + (c3_o3_p_cos_angle - gt_angle) ** 2
    )
    c3_o3_p_loss_per_residue = torch.nn.functional.relu(
        c3_o3_p_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c3_mask * this_o3_mask * next_p_mask * has_no_gap_mask
    c3_o3_p_loss = torch.sum(mask * c3_o3_p_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c3_o3_p_violation_mask = mask * (
        c3_o3_p_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    o3_p_o5_cos_angle = torch.sum((-o3_p_unit_vec) * p_o5_unit_vec, dim=-1)
    gt_angle = nucleotide_constants.between_res_cos_angles_o3_p_o5[0]
    gt_stddev = nucleotide_constants.between_res_cos_angles_o3_p_o5[1]
    o3_p_o5_cos_angle_error = torch.sqrt(
        eps + torch.square(o3_p_o5_cos_angle - gt_angle)
    )
    o3_p_o5_loss_per_residue = torch.nn.functional.relu(
        o3_p_o5_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_o3_mask * next_p_mask * next_o5_mask * has_no_gap_mask
    o3_p_o5_loss = torch.sum(mask * o3_p_o5_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    o3_p_o5_violation_mask = mask * (
        o3_p_o5_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        o3_p_loss_per_residue + c3_o3_p_loss_per_residue + o3_p_o5_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [
                o3_p_violation_mask, 
                c3_o3_p_violation_mask, 
                o3_p_o5_violation_mask
            ],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "o3_p_loss_mean": o3_p_loss,
        "c3_o3_p_loss_mean": c3_o3_p_loss,
        "o3_p_o5_loss_mean": o3_p_o5_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom23_pred_positions: torch.Tensor,
    atom23_atom_exists: torch.Tensor,
    atom23_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 23)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 23)
    """
    fp_type = atom23_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 23, 23)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_pred_positions[..., :, None, :, None, :]
                - atom23_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 23, 23)
    dists_mask = (
        atom23_atom_exists[..., :, None, :, None]
        * atom23_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone O3'--P bond between subsequent residues is no clash.
    o3_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(6), num_classes=23
    )
    o3_one_hot = o3_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *o3_one_hot.shape
    )
    o3_one_hot = o3_one_hot.type(fp_type)
    p_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(8), num_classes=23
    )
    p_one_hot = p_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *p_one_hot.shape
    )
    p_one_hot = p_one_hot.type(fp_type)

    neighbour_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]
    o3_p_bonds = (
        neighbour_mask
        * o3_one_hot[..., None, None, :, None]
        * p_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - o3_p_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 23, 23)
    dists_lower_bound = dists_mask * (
        atom23_atom_radius[..., :, None, :, None]
        + atom23_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 23, 23)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 23)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 23, 23)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 23)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 23)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 23)
    }


def within_residue_violations(
    atom23_pred_positions: torch.Tensor,
    atom23_atom_exists: torch.Tensor,
    atom23_dists_lower_bound: torch.Tensor,
    atom23_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom23_pred_positions ([*, N, 23, 3]):
            Predicted positions of atoms in global prediction frame.
        atom23_atom_exists ([*, N, 23]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom23_dists_lower_bound ([*, N, 23]):
            Lower bound on allowed distances.
        atom23_dists_upper_bound ([*, N, 23]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 23]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 23]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(23, device=atom23_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom23_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom23_atom_exists[..., :, :, None]
        * atom23_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_pred_positions[..., :, :, None, :]
                - atom23_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom23_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom23_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom23_dists_lower_bound) | (dists > atom23_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }



def find_structural_violations(
    atom23_pred_positions: torch.Tensor,
    atom23_exists: torch.Tensor,
    residx_atom23_to_atom27: torch.Tensor,
    butype: torch.Tensor,
    residue_index: torch.Tensor,
    # TODO(yujingcheng): use this asym id
    asym_id: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    butype = correct_rna_butype(butype)
    """Computes several checks for structural violations."""
    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom23_pred_positions,
        pred_atom_mask=atom23_exists,
        residue_index=residue_index,
        nttype=butype,
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        nucleotide_constants.van_der_waals_radius[name[0]]
        for name in nucleotide_constants.atom_types
    ]
    atomtype_radius = atom23_pred_positions.new_tensor(atomtype_radius)
    atom23_atom_radius = (
        atom23_exists
        * atomtype_radius[residx_atom23_to_atom27]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom23_pred_positions=atom23_pred_positions,
        atom23_atom_exists=atom23_exists,
        atom23_atom_radius=atom23_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom23_bounds = nucleotide_constants.make_atom23_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom23_dists_lower_bound = atom23_pred_positions.new_tensor(
        restype_atom23_bounds["lower_bound"]
    )[butype]
    atom23_dists_upper_bound = atom23_pred_positions.new_tensor(
        restype_atom23_bounds["upper_bound"]
    )[butype]
    residue_violations = within_residue_violations(
        atom23_pred_positions=atom23_pred_positions,
        atom23_atom_exists=atom23_exists,
        atom23_dists_lower_bound=atom23_dists_lower_bound,
        atom23_dists_upper_bound=atom23_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_o3_p_loss_mean": connection_violations["o3_p_loss_mean"],  # ()
            "angles_c3_o3_p_loss_mean": connection_violations[
                "c3_o3_p_loss_mean"
            ],  # ()
            "angles_o3_p_o5_loss_mean": connection_violations[
                "o3_p_o5_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N,20)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 20)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 20)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 20),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def violation_loss(
    data,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    num_atoms = torch.sum(data["dense_atom_exists"])

    violations = find_structural_violations(
        data["pred_dense_positions"][-1],
        data["dense_atom_exists"],
        data["residx_dense_to_all"],
        data["butype"],
        data["residue_index"],
        data["asym_id"],
        violation_tolerance_factor,
        clash_overlap_tolerance,
    )    

    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"]
    )
    l_clash = l_clash / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_o3_p_loss_mean"]
        + violations["between_residues"]["angles_c3_o3_p_loss_mean"]
        + violations["between_residues"]["angles_o3_p_o5_loss_mean"]
        + l_clash
    )

    loss = loss.mean()
    
    return loss


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
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

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    butype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [8, *, N, 4, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            butype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 4] angle mask
            chi_angles_sin_cos:
                [*, N, 4, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos
    butype = correct_rna_butype(butype)
    residue_type_one_hot = torch.nn.functional.one_hot(
        torch.clamp(butype, max=nucleotide_constants.restype_num),
        nucleotide_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(nucleotide_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )
    sq_chi_loss = masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def geometry_loss(
    data,
) -> torch.Tensor:
    loss = 0
    # The keys can be read from config file
    dis_keys = ['dis_n', 'dis_c4', 'dis_p', 'omg', 'theta']
    for key in dis_keys:
        loss += softmax_cross_entropy(data["pred_" + key], data[key])
    
    # Average over the batch dimension
    loss = torch.mean(loss)
    loss /= len(dis_keys)
    return loss


def torsion_loss(
    data,
) -> torch.Tensor:
    dis_keys = ['eta_bb', 'theta_bb', 'chi']
    # The keys can be read from config file
    loss = softmax_cross_entropy(
        data['angles_dis'],
        torch.stack([data[key] for key in dis_keys], dim=-2)
    )
    
    # Average over the batch dimension
    loss = torch.mean(loss)
    loss /= len(dis_keys)
    return loss


def disbackbone_loss(
    data,
    eps=1e-8,
    **kwargs,
):
    atom23_pred_positions = data["pred_dense_positions"][-1]
    atom23_gt_positions = data["dense_atom_gt_positions"]
    atom23_mask = data["dense_atom_gt_exists"]
    # C2'-C3'
    pred_c2c3 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_pred_positions[..., 0, :] * atom23_mask[..., 0, None]
                - atom23_pred_positions[..., 3, :] * atom23_mask[..., 3, None]
            )
            ** 2,
            dim=-1,
        )
    )
    gt_c2c3 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_gt_positions[..., 0, :] * atom23_mask[..., 0, None]
                - atom23_gt_positions[..., 3, :] * atom23_mask[..., 3, None]
            )
            ** 2,
            dim=-1,
        )
    )
    # O4'-C1'
    pred_o4c1 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_pred_positions[..., 2, :] * atom23_mask[..., 2, None]
                - atom23_pred_positions[..., 4, :] * atom23_mask[..., 4, None]
            )
            ** 2,
            dim=-1,
        )
    )
    gt_o4c1 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_gt_positions[..., 2, :] * atom23_mask[..., 2, None]
                - atom23_gt_positions[..., 4, :] * atom23_mask[..., 4, None]
            )
            ** 2,
            dim=-1,
        )
    )
    # C4'-C1'
    pred_c4c1 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_pred_positions[..., 1, :] * atom23_mask[..., 1, None]
                - atom23_pred_positions[..., 4, :] * atom23_mask[..., 4, None]
            )
            ** 2,
            dim=-1,
        )
    )
    gt_c4c1 = torch.sqrt(
        eps
        + torch.sum(
            (
                atom23_gt_positions[..., 1, :] * atom23_mask[..., 1, None]
                - atom23_gt_positions[..., 4, :] * atom23_mask[..., 4, None]
            )
            ** 2,
            dim=-1,
        )
    )
    # # C4'-C2'
    # pred_c4c2 = torch.sqrt(
    #     eps
    #     + torch.sum(
    #         (
    #             atom23_pred_positions[..., 1, :] * atom23_mask[..., 1, None]
    #             - atom23_pred_positions[..., 3, :] * atom23_mask[..., 3, None]
    #         )
    #         ** 2,
    #         dim=-1,
    #     )
    # )
    # gt_c4c2 = torch.sqrt(
    #     eps
    #     + torch.sum(
    #         (
    #             atom23_gt_positions[..., 1, :] * atom23_mask[..., 1, None]
    #             - atom23_gt_positions[..., 3, :] * atom23_mask[..., 3, None]
    #         )
    #         ** 2,
    #         dim=-1,
    #     )
    # )
    
    # errors = torch.abs(pred_c2c3 - gt_c2c3) + torch.abs(pred_o4c1 - gt_o4c1) + torch.abs(pred_c4c1 - gt_c4c1) + torch.abs(pred_c4c2 - gt_c4c2)
    errors = torch.abs(pred_c2c3 - gt_c2c3) + torch.abs(pred_o4c1 - gt_o4c1) + torch.abs(pred_c4c1 - gt_c4c1)
    
    loss = torch.mean(errors)
    loss /= 3
    return loss


def residue_angle_loss(
    data,
    eps=1e-8,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    **kwargs,
):
    
    atom23_pred_positions = data["pred_dense_positions"][-1]
    atom23_mask = data["dense_atom_gt_exists"]

    c3_pos = atom23_pred_positions[..., 0, :]
    c3_mask = atom23_mask[..., 0]
    c4_pos = atom23_pred_positions[..., 1, :]
    c4_mask = atom23_mask[..., 1]
    o4_pos = atom23_pred_positions[..., 2, :]
    o4_mask = atom23_mask[..., 2]
    c2_pos = atom23_pred_positions[..., 3, :]
    c2_mask = atom23_mask[..., 3]
    c1_pos = atom23_pred_positions[..., 4, :]
    c1_mask = atom23_mask[..., 4]
    n_pos = atom23_pred_positions[..., 11, :]
    n_mask = atom23_mask[..., 11]
    # c_pos = atom23_pred_positions[..., 13, :]
    # c_mask = atom23_mask[..., 13]
    
    ################################ C1'-N-C base ################################
    # Compute loss for the angles.
    n_c1_bond_length = torch.sqrt(
        eps + torch.sum((c1_pos - n_pos) ** 2, dim=-1)
    )
    # n_c_bond_length = torch.sqrt(
    #     eps + torch.sum((n_pos - c_pos) ** 2, dim=-1)
    # )

    # n_c1_unit_vec = (c1_pos - n_pos) / n_c1_bond_length[..., None]
    # n_c_unit_vec = (c_pos - n_pos) / n_c_bond_length[..., None]
    
    # c1_n_c_cos_angle = torch.sum(n_c1_unit_vec * n_c_unit_vec, dim=-1)
    # gt_angle = nucleotide_constants.between_res_cos_angles_c1_n_c[0]
    # gt_stddev = nucleotide_constants.between_res_cos_angles_c1_n_c[1]
    # c1_n_c_cos_angle_error = torch.sqrt(
    #     eps + (c1_n_c_cos_angle - gt_angle) ** 2
    # )
    # c1_n_c_loss_per_residue = torch.nn.functional.relu(
    #     c1_n_c_cos_angle_error - tolerance_factor_soft * gt_stddev
    # )
    # mask = c1_mask * n_mask * c_mask
    # c1_n_c_loss = torch.sum(mask * c1_n_c_loss_per_residue, dim=-1) / (
    #     torch.sum(mask, dim=-1) + eps
    # )
    # c1_n_c_violation_mask = mask * (
    #     c1_n_c_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    # )
    
    ################################ O4'-C1'-N ################################
    c1_o4_bond_length = torch.sqrt(
        eps + torch.sum((o4_pos - c1_pos) ** 2, dim=-1)
    )
    c1_o4_unit_vec = (o4_pos - c1_pos) / c1_o4_bond_length[..., None]
    c1_n_unit_vec = (n_pos - c1_pos) / n_c1_bond_length[..., None]
    
    o4_c1_n_cos_angle = torch.sum(c1_o4_unit_vec * c1_n_unit_vec, dim=-1)
    gt_angle = nucleotide_constants.between_res_cos_angles_o4_c1_n[0]
    gt_stddev = nucleotide_constants.between_res_cos_angles_o4_c1_n[1]
    o4_c1_n_cos_angle_error = torch.sqrt(
        eps + (o4_c1_n_cos_angle - gt_angle) ** 2
    )
    o4_c1_n_loss_per_residue = torch.nn.functional.relu(
        o4_c1_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = o4_mask * c1_mask * n_mask
    o4_c1_n_loss = torch.sum(mask * o4_c1_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    o4_c1_n_violation_mask = mask * (
        o4_c1_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )
    
    ################################ C3'-C2'-C1' ################################
    c1_c2_bond_length = torch.sqrt(
        eps + torch.sum((c1_pos - c2_pos) ** 2, dim=-1)
    )
    c3_c2_bond_length = torch.sqrt(
        eps + torch.sum((c3_pos - c2_pos) ** 2, dim=-1)
    )
    c2_c1_unit_vec = (c1_pos - c2_pos) / c1_c2_bond_length[..., None]
    c2_c3_unit_vec = (c3_pos - c2_pos) / c3_c2_bond_length[..., None]
    
    c1_c2_c3_cos_angle = torch.sum(c2_c1_unit_vec * c2_c3_unit_vec, dim=-1)
    gt_angle = nucleotide_constants.between_res_cos_angles_c1_c2_c3[0]
    gt_stddev = nucleotide_constants.between_res_cos_angles_c1_c2_c3[1]
    c1_c2_c3_cos_angle_error = torch.sqrt(
        eps + (c1_c2_c3_cos_angle - gt_angle) ** 2
    )
    c1_c2_c3_loss_per_residue = torch.nn.functional.relu(
        c1_c2_c3_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = c1_mask * c2_mask * c3_mask
    c1_c2_c3_loss = torch.sum(mask * c1_c2_c3_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c1_c2_c3_violation_mask = mask * (
        c1_c2_c3_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )
    
    ################################ C4'-C3'-C2' ################################
    c4_c3_bond_length = torch.sqrt(
        eps + torch.sum((c4_pos - c3_pos) ** 2, dim=-1)
    )
    c3_c2_unit_vec = (c2_pos - c3_pos) / c3_c2_bond_length[..., None]
    c3_c4_unit_vec = (c4_pos - c3_pos) / c4_c3_bond_length[..., None]
    
    c2_c3_c4_cos_angle = torch.sum(c3_c2_unit_vec * c3_c4_unit_vec, dim=-1)
    gt_angle = nucleotide_constants.between_res_cos_angles_c2_c3_c4[0]
    gt_stddev = nucleotide_constants.between_res_cos_angles_c2_c3_c4[1]
    c2_c3_c4_cos_angle_error = torch.sqrt(
        eps + (c2_c3_c4_cos_angle - gt_angle) ** 2
    )
    c2_c3_c4_loss_per_residue = torch.nn.functional.relu(
        c2_c3_c4_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = c2_mask * c3_mask * c4_mask
    c2_c3_c4_loss = torch.sum(mask * c2_c3_c4_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c2_c3_c4_violation_mask = mask * (
        c2_c3_c4_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    loss = o4_c1_n_loss + c1_c2_c3_loss + c2_c3_c4_loss
    
    loss /= 3
    loss = torch.mean(loss)
    return loss


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom23_pred_positions: torch.Tensor,  # (N, 23, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_c4_c4_violations = extreme_c4_c4_distance_violations(
        pred_atom_positions=atom23_pred_positions,
        pred_atom_mask=batch["atom23_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_c4_c4_distance"] = extreme_c4_c4_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def extreme_c4_c4_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_c4_pos = pred_atom_positions[..., :-1, 3, :]
    this_c4_mask = pred_atom_mask[..., :-1, 3]
    next_c4_pos = pred_atom_positions[..., 1:, 3, :]
    next_c4_mask = pred_atom_mask[..., 1:, 3]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    c4_c4_distance = torch.sqrt(
        eps + torch.sum((this_c4_pos - next_c4_pos) ** 2, dim=-1)
    )
    violations = (
        c4_c4_distance - nucleotide_constants.c4_c4
    ) > max_angstrom_tolerance
    mask = this_c4_mask * next_c4_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean
