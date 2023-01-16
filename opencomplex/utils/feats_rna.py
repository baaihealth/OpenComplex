import torch
import torch.nn as nn
from opencomplex.utils.rigid_utils import Rigid, Rotation
from opencomplex.utils.tensor_utils import (
    batched_gather,
)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    butype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    
    # [*, N, 9, 4, 4]
    default_4x4 = rrgdf[butype, ...]

    # [*, N, 9] transformations, i.e.
    #   One [*, N, 9, 3, 3] rotation matrix and
    #   One [*, N, 9, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)
    
    bb_rot1 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot1[..., 1] = 1
    
    bb_rot2 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot2[..., 1] = 1

    alpha = torch.cat(
        [
            bb_rot1.expand(*alpha.shape[:-2], -1, -1), 
            bb_rot2.expand(*alpha.shape[:-2], -1, -1), 
            alpha,
        ], dim=-2
    )

    # [*, N, 9, 3, 3]
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
    
    delta_frame_to_frame = all_frames[..., 2]
    gamma_frame_to_frame = all_frames[..., 3]
    beta_frame_to_frame = all_frames[..., 4]
    alpha1_frame_to_frame = all_frames[..., 5]
    alpha2_frame_to_frame = all_frames[..., 6]
    tm_frame_to_frame = all_frames[..., 7]
    chi_frame_to_frame = all_frames[..., 8]

    delta_frame_to_bb = delta_frame_to_frame
    gamma_frame_to_bb = gamma_frame_to_frame
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)
    alpha1_frame_to_bb = beta_frame_to_bb.compose(alpha1_frame_to_frame)
    alpha2_frame_to_bb = beta_frame_to_bb.compose(alpha2_frame_to_frame)
    tm_frame_to_bb = tm_frame_to_frame
    chi_frame_to_bb = chi_frame_to_frame

    all_frames_to_bb = Rigid.cat(
        [ 
            all_frames[..., :2],
            delta_frame_to_bb.unsqueeze(-1),
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha1_frame_to_bb.unsqueeze(-1),
            alpha2_frame_to_bb.unsqueeze(-1),
            tm_frame_to_bb.unsqueeze(-1),
            chi_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )
    
    all_frames_to_global1 = r[..., 0, None].compose(all_frames_to_bb[..., 0:1])
    all_frames_to_global2 = r[..., 1, None].compose(all_frames_to_bb[..., 1:2])
    all_frames_to_global3 = r[..., 0, None].compose(all_frames_to_bb[..., 2:7])
    all_frames_to_global4 = r[..., 1, None].compose(all_frames_to_bb[..., 7:9])
    all_frames_to_global = Rigid.cat(
        [
            all_frames_to_global1, 
            all_frames_to_global2,
            all_frames_to_global3,
            all_frames_to_global4,
        ]
        , dim=-1)

    return all_frames_to_global


def frames_and_literature_positions_to_atom23_pos(
    r: Rigid,
    butype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 23]
    group_mask = group_idx[butype, ...]

    # # [*, N, 23, 9]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 23, 9]
    t_atoms_to_global = r[..., None, :] * group_mask

    # # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 23, 1]
    atom_mask = atom_mask[butype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    lit_positions = lit_positions[butype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions

###########################################################


# get sparse representation of 23 atoms
def atom23_to_atom27_train(atom23, batch):
    atom27_data = batched_gather(
        atom23,
        batch["residx_atom27_to_atom23"],
        dim=-2,
        no_batch_dims=len(atom23.shape[:-2]),
    )

    atom27_data = atom27_data * batch["atom27_atom_exists"][..., None]

    return atom27_data


def atom23_to_atom27_infer(sm, batch):
    
    atom23 = sm['positions'][-1]
    
    atom27_data = batched_gather(
        atom23,
        batch["residx_atom27_to_atom23"][..., 0],
        dim=-2,
        no_batch_dims=len(atom23.shape[:-2]),
    )
    
    atom27_data = atom27_data * batch["atom27_atom_exists"][..., 0:1]
    
    return atom27_data