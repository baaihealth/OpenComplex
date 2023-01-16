import numpy as np
import torch
import torch.nn.functional as F

import enum
class QuaternionCoeffOrder(enum.Enum):
    XYZW = 'xyzw'
    WXYZ = 'wxyz'

## ----------------
## ----------------

class Quaternion:
    def __init__(self, q):
        self._QUAT_MULTIPLY = np.zeros((4, 4, 4))
        self._QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                                        [ 0,-1, 0, 0],
                                        [ 0, 0,-1, 0],
                                        [ 0, 0, 0,-1]]

        self._QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                                        [ 1, 0, 0, 0],
                                        [ 0, 0, 0, 1],
                                        [ 0, 0,-1, 0]]

        self._QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                                        [ 0, 0, 0,-1],
                                        [ 1, 0, 0, 0],
                                        [ 0, 1, 0, 0]]

        self._QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                                        [ 0, 0, 1, 0],
                                        [ 0,-1, 0, 0],
                                        [ 1, 0, 0, 0]]

        self._QUAT_MULTIPLY_BY_VEC = self._QUAT_MULTIPLY[:, 1:, :]

        self.q = q

    def normalize(self):
        self.q = F.normalize(self.q, p=2.0, dim=-1, eps=1.0e-12)

    def to(self, device, dtype):
        self.q.to(device=device, dtype=dtype)
    
    @staticmethod
    def quat_multiply(quat1, quat2):
        """Multiply a quaternion by another quaternion."""
        mat = quat1.new_tensor(Quaternion._QUAT_MULTIPLY)
        reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
        rt = torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))
        return rt

    @staticmethod
    def quat_multiply_by_vec(quat, vec):
        """Multiply a quaternion by a pure-vector (w=0) quaternion."""
        mat = quat.new_tensor(Quaternion._QUAT_MULTIPLY_BY_VEC)
        reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
        rt = torch.sum(reshaped_mat*quat[..., :, None, None]*vec[..., None, :, None], dim=(-3, -2))
        return rt

## ----------------
## ----------------


class AffineTransformation:
    def __init__(self, rots=None, trans=None):
        self.r = rots
        self.t = trans
        if self.t is None:
            self.t = torch.zeros_like(self.r)[...,0]

    def __mul__(self, other):
        if type(other) == AffineTransformation:
            r1 = self.r
            t1 = self.t
            r2 = other.r
            t2 = other.t

            if t2 is None:
                t2 = torch.zeros_like(t1)

            new_rot = torch.matmul(r1, r2)
            new_trans = torch.matmul(r1, t2[..., None]).squeeze(-1) + t1
            
            rt = AffineTransformation(new_rot, new_trans)

        elif type(other) == torch.Tensor:
            new_rot = self.r * other[..., None, None]
            new_trans = self.t * other[..., None]

            rt = AffineTransformation(new_rot, new_trans)

        return rt
    
    def __getitem__(self, index):
        if type(index) != tuple:
            index = (index,)

        rt = AffineTransformation(self.r[index + (slice(None), slice(None))], self.t[index + (slice(None),)])
        return rt
    
    @staticmethod
    def compose_rot(r1, r2):
        r = torch.matmul(r1, r2)
        return r

    @staticmethod
    def apply_rot(r, p):
        p = torch.matmul(r, p[..., None]).squeeze(-1)
        return p


    @staticmethod
    def cat(ts, dim):
        rots = torch.cat([t.r for t in ts], dim=dim if dim >= 0 else dim - 2) 
        trans = torch.cat([t.t for t in ts], dim=dim if dim >= 0 else dim - 1)

        return AffineTransformation(rots, trans)

    @staticmethod
    def from_3_points_pos(p_x_axis, origin, p_xy_plane, eps=1e-4):
        e0 = p_x_axis - origin
        e0 = e0/(e0.norm(dim=-1, keepdim=True) + eps)

        e1 = p_xy_plane - origin
        dot = (e0*e1).sum(dim=-1, keepdim=True)
        e1 = e1 - e0*dot
        e1 = e1/(e1.norm(dim=-1, keepdim=True) + eps)

        e2 = torch.cross(e0, e1, dim=-1)

        rots = torch.stack([e0,e1,e2], dim=-1)
        t = origin

        return AffineTransformation(rots, t)

    @staticmethod
    def from_3_points(p_neg_x_axis, origin, p_xy_plane, eps=1e-8):
        e0 = origin - p_neg_x_axis
        e0 = e0/e0.norm(dim=-1, keepdim=True)

        e1 = p_xy_plane - origin
        dot = (e0*e1).sum(dim=-1, keepdim=True)
        e1 = e1 - e0*dot
        e1 = e1/e1.norm(dim=-1, keepdim=True)

        e2 = torch.cross(e0, e1, dim=-1)

        rots = torch.stack([e0,e1,e2], dim=-1)
        t = origin

        return AffineTransformation(rots, t)

    def sum(self, dim):
        if dim >= len(self.r.shape)-2:
            raise ValueError("Invalid dimension")

        r = self.r.sum(dim if dim >= 0 else dim - 2)
        t = self.t.sum(dim if dim >= 0 else dim - 1)

        return AffineTransformation(r, t)

    @staticmethod
    def apply_affine(aff, p):
        rotated = AffineTransformation.apply_rot(aff.r, p)
        rt = rotated + aff.t
        return rt

    def invert(self):
        rot_inv = self.r.transpose(-1, -2)
        trn_inv = AffineTransformation.apply_rot(rot_inv, self.t)

        return AffineTransformation(rot_inv, -1 * trn_inv)

    def to(self, device, dtype):
        self.r.to(device=device, dtype=dtype)
        self.t.to(device=device, dtype=dtype)
    
    def unsqueeze(self, dim):
        if dim >= len(self.r.shape)-2:
            raise ValueError("Invalid dimension")
        r = self.r.unsqueeze(dim if dim >= 0 else dim - 2)
        t = self.t.unsqueeze(dim if dim >= 0 else dim - 1)

        return AffineTransformation(r, t)
    
    @staticmethod
    def from_tensor_4x4(t):
        if(t.shape[-2:] != (4, 4)):
            raise ValueError("Incorrectly shaped input tensor")

        rots = t[..., :3, :3]
        trans = t[..., :3, 3]
        
        return AffineTransformation(rots, trans)

    def to_tensor_4x4(self):
        shape = self.r.shape[:-2]
        tensor = self.t.new_zeros((*shape, 4, 4))
        tensor[..., :3, :3] = self.r
        tensor[..., :3, 3] = self.t
        tensor[..., 3, 3] = 1

        return tensor
    
    def contiguous(self):
        r = self.r.contiguous()
        t = self.t.contiguous()

        return AffineTransformation(r, t)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = torch.matmul(c2_rots, c1_rots)
        n_xyz = AffineTransformation.apply_rot(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = torch.matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        r = AffineTransformation(rots, translation)

        return r 

    @staticmethod
    def apply_invert_affine(aff, pts):
        pts = pts - aff.t
        rot_inv = aff.r.transpose(-1, -2)
        r = AffineTransformation.apply_rot(rot_inv, pts)
        return r

    