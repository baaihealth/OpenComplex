# Unit test for AffineTransformation

import math
import numpy as np
import torch
import unittest

from opencomplex.utils.rigid_utils import Rotation, Rigid, quat_to_rot, rot_to_quat
from opencomplex.utils.kornia_utils import AffineTransformation

from opencomplex.utils.tensor_utils import chunk_layer, _chunk_slice
import tests.compare_utils as compare_utils
from tests.config import consts

X_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)

X_NEG_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)


class TestUtils(unittest.TestCase):
    def test_rigid_from_3_points_shape(self):
        batch_size = 2
        n_res = 5

        x1 = torch.rand((batch_size, n_res, 3))
        x2 = torch.rand((batch_size, n_res, 3))
        x3 = torch.rand((batch_size, n_res, 3))

        r = Rigid.from_3_points(x1, x2, x3)

        rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

        self.assertTrue(rot.shape == (batch_size, n_res, 3, 3))
        self.assertTrue(torch.all(tra == x2))
        
        ## ----------
        r_a = AffineTransformation.from_3_points(x1, x2, x3)
        self.assertTrue(torch.all(rot == r_a.r))
        self.assertTrue(torch.all(tra == r_a.t))

    def test_rigid_from_4x4(self):
        batch_size = 2
        transf = [
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ]
        transf = torch.tensor(transf)

        true_rot = transf[:3, :3]
        true_trans = transf[:3, 3]

        transf = torch.stack([transf for _ in range(batch_size)], dim=0)

        r = Rigid.from_tensor_4x4(transf)

        rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

        self.assertTrue(torch.all(rot == true_rot.unsqueeze(0)))
        self.assertTrue(torch.all(tra == true_trans.unsqueeze(0)))

        ## ----------
        r_a = AffineTransformation.from_tensor_4x4(transf)
        self.assertTrue(torch.all(rot == r_a.r))
        self.assertTrue(torch.all(tra == r_a.t))

        self.assertTrue(torch.all(r_a.to_tensor_4x4() == r.to_tensor_4x4()))

    def test_rigid_shape(self):
        batch_size = 2
        n = 5
        rot_rand = torch.rand((batch_size, n, 3, 3))
        tra_rand = torch.rand((batch_size, n, 3))
        transf = Rigid(
            Rotation(rot_mats=rot_rand), 
            tra_rand
        )

        self.assertTrue(transf.shape == (batch_size, n))

        ## ----------
        r_a = AffineTransformation(rot_rand, tra_rand)
        self.assertTrue(r_a.r.shape == (batch_size, n, 3, 3))
        self.assertTrue(r_a.t.shape == (batch_size, n, 3))

    def test_rigid_cat(self):
        batch_size = 2
        n = 5
        rot_rand = torch.rand((batch_size, n, 3, 3))
        tra_rand = torch.rand((batch_size, n, 3))
        transf = Rigid(
            Rotation(rot_mats=rot_rand), 
            tra_rand
        )

        transf_cat = Rigid.cat([transf, transf], dim=0)

        transf_rots = transf.get_rots().get_rot_mats()
        transf_cat_rots = transf_cat.get_rots().get_rot_mats()

        self.assertTrue(transf_cat_rots.shape == (batch_size * 2, n, 3, 3))

        transf_cat = Rigid.cat([transf, transf], dim=1)
        transf_cat_rots = transf_cat.get_rots().get_rot_mats()

        self.assertTrue(transf_cat_rots.shape == (batch_size, n * 2, 3, 3))

        self.assertTrue(torch.all(transf_cat_rots[:, :n] == transf_rots))
        self.assertTrue(
            torch.all(transf_cat.get_trans()[:, :n] == transf.get_trans())
        )

        ## ----------
        r_a = AffineTransformation(rot_rand, tra_rand)
        r_a_cat = AffineTransformation.cat([r_a, r_a], dim=0)
        r_a_cat_rot = r_a_cat.r
        self.assertTrue(r_a_cat_rot.shape == (batch_size * 2, n, 3, 3))

        r_a_cat = AffineTransformation.cat([r_a, r_a], dim=1)
        r_a_cat_rot = r_a_cat.r
        self.assertTrue(r_a_cat_rot.shape == (batch_size, n * 2, 3, 3))

        self.assertTrue(torch.all(r_a_cat_rot[:, :n] == r_a.r))
        self.assertTrue(torch.all(r_a_cat.t[:, :n] == r_a.t))

    def test_mul(self):
        batch_size = 2
        n = 5
        rot_rand = torch.rand((batch_size, n, 3, 3))
        tra_rand = torch.rand((batch_size, n, 3))
        transf = Rigid(
            Rotation(rot_mats=rot_rand), 
            tra_rand
        )

        mul_r = torch.rand((batch_size, n))
        transf_mul = transf*mul_r

        ## ----------
        r_a = AffineTransformation(rot_rand, tra_rand)
        r_a_mul = r_a*mul_r
       
        self.assertTrue(torch.all(transf_mul.get_rots().get_rot_mats() == r_a_mul.r))
        self.assertTrue(torch.all(transf_mul.get_trans() == r_a_mul.t))

    def test_invert(self):
        batch_size = 2
        n = 5
        rot_rand = torch.rand((batch_size, n, 3, 3))
        tra_rand = torch.rand((batch_size, n, 3))
        transf = Rigid(
            Rotation(rot_mats=rot_rand), 
            tra_rand
        )

        transf_inv = transf.invert()

        ## ----------
        r_a = AffineTransformation(rot_rand, tra_rand)
        r_a_inv = r_a.invert()
       
        self.assertTrue(torch.all(transf_inv.get_rots().get_rot_mats() == r_a_inv.r))
        self.assertTrue(torch.all(transf_inv.get_trans() == r_a_inv.t))

    def test_rigid_compose(self):
        trans_1 = [0, 1, 0]
        trans_2 = [0, 0, 1]

        t1 = Rigid(
            Rotation(rot_mats=X_90_ROT), 
            torch.tensor(trans_1)
        )
        t2 = Rigid(
            Rotation(rot_mats=X_NEG_90_ROT), 
            torch.tensor(trans_2)
        )

        t3 = t1.compose(t2)

        self.assertTrue(
            torch.all(t3.get_rots().get_rot_mats() == torch.eye(3))
        )
        self.assertTrue(
            torch.all(t3.get_trans() == 0)
        )

        ## ----------
        r_a1 = AffineTransformation(X_90_ROT, torch.tensor(trans_1))
        r_a2 = AffineTransformation(X_NEG_90_ROT, torch.tensor(trans_2))
        r_a3 = r_a1*r_a2
        self.assertTrue(torch.all(r_a3.r == torch.eye(3)))
        self.assertTrue(torch.all(r_a3.t == 0))

    def test_rigid_apply(self):
        rots = torch.stack([X_90_ROT, X_NEG_90_ROT], dim=0)
        trans = torch.tensor([1, 1, 1])
        trans = torch.stack([trans, trans], dim=0)

        t = Rigid(Rotation(rot_mats=rots), trans)

        x = torch.arange(30)
        x = torch.stack([x, x], dim=0)
        x = x.view(2, -1, 3)  # [2, 10, 3]

        pts = t[..., None].apply(x)

        # All simple consequences of the two x-axis rotations
        self.assertTrue(torch.all(pts[..., 0] == x[..., 0] + 1))
        self.assertTrue(torch.all(pts[0, :, 1] == x[0, :, 2] * -1 + 1))
        self.assertTrue(torch.all(pts[1, :, 1] == x[1, :, 2] + 1))
        self.assertTrue(torch.all(pts[0, :, 2] == x[0, :, 1] + 1))
        self.assertTrue(torch.all(pts[1, :, 2] == x[1, :, 1] * -1 + 1))

        ## ----------
        r_a = AffineTransformation(rots, trans)
        pts = AffineTransformation.apply_affine(r_a[..., None], x)

        self.assertTrue(torch.all(pts[..., 0] == x[..., 0] + 1))
        self.assertTrue(torch.all(pts[0, :, 1] == x[0, :, 2] * -1 + 1))
        self.assertTrue(torch.all(pts[1, :, 1] == x[1, :, 2] + 1))
        self.assertTrue(torch.all(pts[0, :, 2] == x[0, :, 1] + 1))
        self.assertTrue(torch.all(pts[1, :, 2] == x[1, :, 1] * -1 + 1))

    def test_rigid_invert_apply(self):
        rots = torch.stack([X_NEG_90_ROT, X_90_ROT], dim=0)
        trans = torch.tensor([1, 1, 1])
        trans = torch.stack([trans, trans], dim=0)

        t = Rigid(Rotation(rot_mats=rots), trans)

        x = torch.arange(30)
        x = torch.stack([x, x], dim=0)
        x = x.view(2, -1, 3)  # [2, 10, 3]

        pts = t[..., None].invert_apply(x)

        # All simple consequences of the two x-axis rotations
        self.assertTrue(torch.all(pts[..., 0] == x[..., 0] - 1))
        self.assertTrue(torch.all(pts[0, :, 1] == (x[0, :, 2] - 1)*-1 ))
        self.assertTrue(torch.all(pts[1, :, 1] == x[1, :, 2] - 1))
        self.assertTrue(torch.all(pts[0, :, 2] == x[0, :, 1] - 1))
        self.assertTrue(torch.all(pts[1, :, 2] == (x[1, :, 1] - 1)*-1))

        ## ----------
        r_a = AffineTransformation(rots, trans)
        pts = AffineTransformation.apply_invert_affine(r_a[..., None], x)

        self.assertTrue(torch.all(pts[..., 0] == x[..., 0] - 1))
        self.assertTrue(torch.all(pts[0, :, 1] == (x[0, :, 2] - 1)*-1 ))
        self.assertTrue(torch.all(pts[1, :, 1] == x[1, :, 2] - 1))
        self.assertTrue(torch.all(pts[0, :, 2] == x[0, :, 1] - 1))
        self.assertTrue(torch.all(pts[1, :, 2] == (x[1, :, 1] - 1)*-1))

    def test_quat_to_rot(self):
        forty_five = math.pi / 4
        quat = torch.tensor([math.cos(forty_five), math.sin(forty_five), 0, 0])
        rot = quat_to_rot(quat)
        eps = 1e-06
        self.assertTrue(torch.all(torch.abs(rot - X_90_ROT) < eps))

        ## ----------
        r = AffineTransformation.quat_to_rot_mat(quat)
        self.assertTrue(torch.all(torch.abs(r - X_90_ROT) < eps))

    def test_rot_to_quat(self):
        quat = rot_to_quat(X_90_ROT)
        eps = 1e-06
        ans = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0., 0.])
        self.assertTrue(torch.all(torch.abs(quat - ans) < eps))

        ## ----------
        q = AffineTransformation.rot_mat_to_quat(X_90_ROT)
        self.assertTrue(torch.all(torch.abs(q - ans) < eps))


if __name__ == "__main__":
    unittest.main()
