# Unit test for fape loss
 
import torch
import numpy as np
import unittest
from opencomplex.utils.rigid_utils import Rotation, Rigid
from opencomplex.utils.kornia_utils import AffineTransformation

import tests.compare_utils as compare_utils
from tests.config import consts
from tests.data_utils import random_affines_vector, random_affines_4x4
from opencomplex.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    dict_multimap,
)

def affine_vector_to_4x4(affine):
    r = Rigid.from_tensor_7(affine)
    return r.to_tensor_4x4()

class TestLoss(unittest.TestCase):

    def test_run_fape(self):
        batch_size = consts.batch_size
        n_frames = 7
        n_atoms = 5

        x = torch.rand((batch_size, n_atoms, 3))
        x_gt = torch.rand((batch_size, n_atoms, 3))
        rots = torch.rand((batch_size, n_frames, 3, 3))
        rots_gt = torch.rand((batch_size, n_frames, 3, 3))
        trans = torch.rand((batch_size, n_frames, 3))
        trans_gt = torch.rand((batch_size, n_frames, 3))

        ## ------------- Rigid ------------
        from opencomplex.utils.loss import compute_fape
        t = Rigid(Rotation(rot_mats=rots), trans)
        t_gt = Rigid(Rotation(rot_mats=rots_gt), trans_gt)
        frames_mask = torch.randint(0, 2, (batch_size, n_frames)).float()
        positions_mask = torch.randint(0, 2, (batch_size, n_atoms)).float()
        length_scale = 10

        loss = compute_fape(
            pred_frames=t,
            target_frames=t_gt,
            frames_mask=frames_mask,
            pred_positions=x,
            target_positions=x_gt,
            positions_mask=positions_mask,
            length_scale=length_scale,
        )

        ## ------------- Affine ------------
        from opencomplex.utils.loss_xyz import compute_fape
        r_a = AffineTransformation(rots, trans)
        r_a_gt = AffineTransformation(rots_gt, trans_gt)

        loss2 = compute_fape(
            pred_frames=r_a,
            target_frames=r_a_gt,
            frames_mask=frames_mask,
            pred_positions=x,
            target_positions=x_gt,
            positions_mask=positions_mask,
            length_scale=length_scale,
        )

        ## compare
        self.assertTrue(torch.all(loss==loss2))


    def test_tm_loss_compare(self):
        ## not test in initial training
        pass

if __name__ == "__main__":
    unittest.main()