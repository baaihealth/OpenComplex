import unittest

import torch

from opencomplex.utils.rigid_utils import Rotation, Rigid
from opencomplex.utils import permutation

C = 0.7071067811865475
ROT_135 = [
    [-C, -C, 0],
    [C,  -C, 0],
    [0,  0,  1]]
TRANS = [1., 2., 3.]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = Rigid(Rotation(torch.tensor(ROT_135, device=device)), torch.tensor(TRANS, device=device))
N_res = 11
eps = 1e-3

class TestMultichainPermutationAlignment(unittest.TestCase):
    def test_get_transform(self):
        pred = torch.rand(N_res, 3, device=device)
        mask = torch.rand(N_res, device=device) >= 0.1
        gt = R.apply(pred)

        rot, tran = permutation.get_transform(pred, gt, mask, device)

        pred_transformed = gt @ rot + tran
        diff = torch.abs(pred_transformed - pred)

        # The result may have 1e-3 order of error due to tf32 on Ampere GPU
        # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        self.assertTrue(torch.all(diff < eps))

    def test_split_chains(self):
        asym_id = torch.tensor([1,1,1,2,2,2,2,3,4,4,5])
        chains = permutation.split_chains(asym_id)
        true_chains = [(0, 3), (3, 7), (7, 8), (8, 10), (10, 11)]

        self.assertTrue(chains == true_chains)

    def test_perm_to_idx(self):
        chains = [(0, 3), (3, 5), (5, 6), (6, 8)]
        perm = [3, 2, 1, 0]
        idx = permutation.perm_to_idx(perm, chains)
        idx_true = [6, 7, 5, 3, 4, 0, 1, 2]

        self.assertTrue(idx == idx_true)
    
    def test_multichain_permutation_alignment(self):
        batch = {}
        out = {}

        batch_size = 2
        chain_lens = [10, 10, 10, 8, 8, 15, 15]
        entitys = [1, 1, 1, 2, 2, 3, 3]
        syms = [1, 2, 3, 1, 2, 1, 2]
        st = 0
        N_res = sum(chain_lens)
        chains = []
        asym_id = []
        entity_id = []
        sym_id = []
        for i, l in enumerate(chain_lens):
            ed = st + l
            chains.append((st, ed))
            for j in range(l):
                asym_id.append(i + 1)
                entity_id.append(entitys[i])
                sym_id.append(syms[i])
            st = ed

        asym_id = [asym_id] * batch_size
        entity_id = [entity_id] * batch_size
        sym_id = [sym_id] * batch_size

        batch['butype'] = torch.zeros([batch_size, N_res], device=device)
        batch['asym_id'] = torch.tensor(asym_id, device=device)
        batch['sym_id'] = torch.tensor(sym_id, device=device)
        batch['entity_id'] = torch.tensor(entity_id, device=device)

        batch['anchor_asym_id'] = 5

        best_perm = [1, 2, 0, 4, 3, 5, 6]
        best_perm_idx = permutation.perm_to_idx(best_perm, chains)
        best_perm_idx = torch.tensor([best_perm_idx] * batch_size, device=device)

        out['final_atom_positions'] = torch.rand([batch_size, N_res, 37, 3], device=device) * 100
        out['final_atom_mask'] = torch.rand([batch_size, N_res, 37], device=device) >= 0.1

        batch['all_atom_positions'] = permutation.apply_permutation_core(
            R.apply(out['final_atom_positions']), best_perm_idx, dims=-3)

        perm_idx = permutation.multichain_permutation_alignment(batch, out)
        self.assertTrue(torch.all(perm_idx == best_perm_idx))


    def test_apply_permutation_core(self):
        # test permute with 1 dimention
        x = torch.arange(0, 20, device=device).view(2, 1, 5, 2)
        idx = torch.tensor([[1, 2, 0, 4, 3], [3, 0, 2, 4, 1]], device=device)
        y = permutation.apply_permutation_core(x, idx, -2)

        y_true = torch.tensor(
            [[[[2, 3],[4, 5],[0, 1],[8,9],[6,7]]], 
             [[[16,17],[10,11],[14,15],[18,19],[12,13]]]],
            device=device)

        self.assertTrue(torch.all(y == y_true))

        # test permute with 2 dimentions
        x = torch.arange(0, 9, device=device, 
                         dtype=torch.float32).view(1, 3, 3)
        x.requires_grad = True
        idx = torch.tensor([[1, 2, 0]], device=device)
        y = permutation.apply_permutation_core(x, idx, [-1, -2])

        y_true = torch.tensor(
            [[4,5,3],
             [7,8,6],
             [1,2,0]],
            device=device,
            dtype=torch.float32)
        self.assertTrue(torch.all(y == y_true))

        # test if gradient can be proped back to x correctly
        d = y ** 2
        external_grad = torch.arange(1, 10, device=device).view(d.shape)
        d.backward(gradient=external_grad)

        g_x_true = torch.tensor(
            [[0, 14, 32],
             [18, 8, 20],
             [72, 56, 80]],
            dtype=torch.float32,
            device=device
        )

        self.assertTrue(torch.all(g_x_true == x.grad))
        
        
if __name__ == '__main__':
    unittest.main()