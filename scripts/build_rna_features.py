import os
import torch
from torch.nn import functional as F
import numpy as np
import random
import math
from Bio.PDB.vectors import Vector
from Bio.PDB.vectors import calc_dihedral
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from opencomplex.np import nucleotide_constants as nc
from opencomplex.data import mmcif_parsing
from opencomplex.data.data_pipeline import make_mmcif_features

nucleotides = {
        'A': 1,
        'G': 2,
        'C': 3,
        'T': 4,
    }
nt_idx2name = dict(zip(nc.restype_order.values(), nc.restype_order.keys()))
nt_idx2name[4] = 'X'
nt_idx2name_E2E = {
    0: 'X',
    1: 'A',
    2: 'G',
    3: 'C',
    4: 'U',
    5: 'X'
}
N_dict = {'A': 'N9', 'U': 'N1', 'C': 'N1', 'G': 'N9'}

class processing_fas_features():
    def read_hmm(self, hmm_file, L):
        count = 0
        seq_idx = 0
        hmm = np.zeros([L,5,3])
        f = open(hmm_file,'r')
        lines = f.readlines()
        for line in lines[int(-L*3-1):-1]:
            line = line.strip()
            line = line.split()
            if(count==0):
                try:
                    hmm[seq_idx,0,0] = float(line[1])
                except:
                    pass
                try:
                    hmm[seq_idx,1,0] = float(line[2])
                except:
                    pass
                try:
                    hmm[seq_idx,2,0] = float(line[3])
                except:
                    pass
                try:
                    hmm[seq_idx,3,0] = float(line[4])
                except:
                    pass
            elif(count==1):
                try:
                    hmm[seq_idx,0,1] = float(line[0])
                except:
                    pass
                try:
                    hmm[seq_idx,1,1] = float(line[1])
                except:
                    pass
                try:
                    hmm[seq_idx,2,1] = float(line[2])
                except:
                    pass
                try:
                    hmm[seq_idx,3,1] = float(line[3])
                except:
                    pass
            elif(count==2):
                try:
                    hmm[seq_idx,0,2] = float(line[0])
                except:
                    pass
                try:
                    hmm[seq_idx,1,2] = float(line[1])
                except:
                    pass
                try:
                    hmm[seq_idx,2,2] = float(line[2])
                except:
                    pass
                try:
                    hmm[seq_idx,3,2] = float(line[3])
                except:
                    pass
                try:
                    hmm[seq_idx,4,0] = float(line[4])
                except:
                    pass
                try:
                    hmm[seq_idx,4,1] = float(line[5])
                except:
                    pass
                try:
                    hmm[seq_idx,4,2] = float(line[6])
                except:
                    pass
            count+=1
            if(count>2):
                count = 0
                seq_idx+=1
        hmm = torch.from_numpy(hmm.reshape(L,15)).float()

        return hmm


    def read_msa(self, msa_file):
        f = open(msa_file,'r')
        lines = f.read()
        f.close()

        lines = lines.split('\n')
        lines = lines[1::2]

        msa_ = np.array([list(s.strip()) for s in lines])
        msa = np.zeros_like(msa_,dtype=int)
        for akey in list(nucleotides.keys()):
            msa[msa_==akey]=nucleotides[akey]

        return msa


    def read_seq(self, seq_file):
        sequence_lines=open(seq_file).readlines()
        sequence = sequence_lines[1].strip()
        L = len(sequence)
        seq_array = np.zeros(L)
        sequence_list = np.array(list(sequence))
        seq_array[sequence_list=='A']=1
        seq_array[sequence_list=='G']=2
        seq_array[sequence_list=='C']=3
        seq_array[sequence_list=='U']=4

        return seq_array


    def randomly_sample_msa(self, unsampled_msa):
        msa_rand = unsampled_msa[1:,:]
        num_seqs,length = msa_rand.shape
        if num_seqs>=1:
            np.random.shuffle(msa_rand)
            num_sel = min(int(num_seqs), 128)
            idx = np.random.randint(num_seqs,size=num_sel)
            msa = msa_rand[idx,:]
        else:
            msa = np.int64(unsampled_msa)

        return msa


    def collect_features(self, seq_file, msa_file, hmm_file, ss_file):
        seq = self.read_seq(seq_file)
        msa = self.read_msa(msa_file)
        msa = self.randomly_sample_msa(msa)
        msa[0] = seq[None,:]
        msa = torch.from_numpy(msa).long()
        num_seqs, length = msa.shape
        if num_seqs<1.9:
            msa = torch.cat([msa,msa],0)
        msa = F.one_hot(msa,6)

        hmm = None
        if(os.path.exists(hmm_file)):
            hmm = self.read_hmm(hmm_file,length)
        else:
            hmm = torch.zeros([length,15])

        ss = None
        if(os.path.exists(ss_file)):
            ss = np.loadtxt(ss_file,skiprows=1)[:-1,:]
        else:
            ss = np.zeros([length,length])

        ss = torch.from_numpy(ss).float()[...,None]

        features = {}
        num_bu = msa[0].shape[0]
        features["ss"] = ss.numpy()
        features["msa"] = msa.to(torch.int64).numpy()
        features["seq"] = msa[0].float().numpy()
        features["hmm"] = hmm.float().numpy()
        features['butype'] = torch.argmax(msa[0], dim=-1).numpy()
        features['residue_index'] = np.array(range(num_bu), dtype=np.int32)
        features['seq_length'] = np.array([num_bu] * num_bu, dtype=np.int32)

        return features

class processing_cif_features():
    def parse_mmcif(self, path, file_id, chain_id=None, butype=None):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if (mmcif_object.mmcif_object is None):
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object
        seq_fa = ''.join([nt_idx2name_E2E[idx] for idx in butype])

        if chain_id is not None:
            data = make_mmcif_features(
                mmcif_object=mmcif_object,
                chain_id=chain_id,
                chain_type=2,
            )
            seq_cif = ''.join([nt_idx2name[idx.item()] for idx in np.argmax(data['butype'], axis=1)])
            if len(seq_fa) != len(seq_cif):
                return None
            for i, j in zip(seq_fa, seq_cif):
                if i != "X" and j != "X" and i != j:
                    return None
            return data
        else:
            chain_ids = list(mmcif_object.chain_to_seqres.keys())
            for chain_id in chain_ids:
                seq_cif = mmcif_object.chain_to_seqres[chain_id]
                if len(seq_fa) != len(seq_cif):
                    continue
                for i, j in zip(seq_fa, seq_cif):
                    if i != "X" and j != "X" and i != j:
                        continue
                data = make_mmcif_features(
                    mmcif_object=mmcif_object,
                    chain_id=chain_id,
                    chain_type=2,
                )
                data['chain_id_new'] = chain_id
                return data
            return None

    def processing_features(self, data):
        features = {}
        seq_length = data['butype'].shape[0]
        butpye_idx = np.argmax(data['butype'], axis=1)
        for nt_idx in range(seq_length):
            butype = nc.restypes_with_x[butpye_idx[nt_idx]]
            nt_key = str(nt_idx) + ',' + butype
            features[nt_key] = {}
            for atom_idx in range(27):
                atom_pos = list(data['all_atom_positions'][nt_idx][atom_idx])
                atom_type = nc.atom_types[atom_idx]
                if np.sum(np.abs(data['all_atom_positions'][nt_idx][atom_idx])) > 0:
                    features[nt_key][atom_type] = [float(item) for item in atom_pos]

        return features

    def get_euc_dis(self, pos_A, pos_B):
        tensorA = torch.tensor(pos_A)
        tensorB = torch.tensor(pos_B)

        return np.linalg.norm(tensorA - tensorB, ord=2)

    def chain_to_sorted_id(self, chain):
        seq = []
        for key in chain.keys():
            seq_id, base = key.split(',')
            seq.append((seq_id, base))

        seq_dict = dict(seq)
        dict_sorted = sorted(seq_dict.items(), key=lambda d: int(d[0]))
        sorted_id = [seq_item[0] + ',' + seq_item[1] for seq_item in dict_sorted]

        return sorted_id

    def dist_to_onehot(self, dist):
        if dist < 2:
            onehot = 0
        elif dist > 40:
            onehot = 39
        else:
            onehot = round(dist) - 1

        return onehot

    def ori_to_onehot(self, ori, dist):
        unit = math.pi / 12
        if dist > 40:
            onehot = 24
        else:
            onehot = round(ori // unit)

        return onehot

    def get_orientation(self, pos1, pos2, pos3, pos4):
        angle_dihedral = calc_dihedral(Vector(pos1), Vector(pos2), Vector(pos3), Vector(pos4))
        return angle_dihedral

    def get_pseudo_torsion_matrix(self, chain, L, l_sorted_nt):
        unit = math.pi / 12
        eta_bb_matrix = np.zeros((L, 24))
        theta_bb_matrix = np.zeros((L, 24))
        eta_bb_matrix[0, 0] = eta_bb_matrix[L - 1, 0] = theta_bb_matrix[0, 0] = theta_bb_matrix[L - 1, 0] = 1

        for i in range(1, L - 1):
            nt = l_sorted_nt[i]
            nt_prev = l_sorted_nt[i - 1]
            nt_next = l_sorted_nt[i + 1]
            try:
                C_im1 = chain[nt_prev]["C4'"]
                P_i = chain[nt]['P']
                C_i = chain[nt]["C4'"]
                P_ip1 = chain[nt_next]['P']

                eta = self.get_orientation(C_im1, P_i, C_i, P_ip1)
                onehot_eta = round(eta // unit)
                eta_bb_matrix[i, onehot_eta] = 1
            except:
                pass
            try:
                C_ip1 = chain[nt_next]["C4'"]
                theta = self.get_orientation(P_i, C_i, P_ip1, C_ip1)
                onehot_theta = round(theta // unit)
                theta_bb_matrix[i, onehot_theta] = 1
            except:
                pass

        return eta_bb_matrix, theta_bb_matrix

    def get_dist_angle_matrix(self, chain, L, l_sorted_nt):
        dist_P_matrix = np.zeros((L, L, 40))
        dist_N_matrix = np.zeros((L, L, 40))
        dist_C_matrix = np.zeros((L, L, 40))
        ori_omg_matrix = np.zeros((L, L, 25))
        ori_theta_matrix = np.zeros((L, L, 25))

        l_sorted_nt = self.chain_to_sorted_id(chain)
        for i in range(L):
            for j in range(L):
                try:
                    P_i = chain[l_sorted_nt[i]]['P']
                    P_j = chain[l_sorted_nt[j]]['P']
                    dist_P = get_euc_dis(P_i, P_j)
                    onehot_P = self.dist_to_onehot(dist_P)
                    dist_P_matrix[i, j, onehot_P] = 1
                except:
                    pass
                try:
                    C_i = chain[l_sorted_nt[i]]["C4'"]
                    C_j = chain[l_sorted_nt[j]]["C4'"]
                    dist_C = self.get_euc_dis(C_i, C_j)
                    onehot_C = self.dist_to_onehot(dist_C)
                    dist_C_matrix[i, j, onehot_C] = 1
                except:
                    pass

                try:
                    N_i = chain[l_sorted_nt[i]][N_dict[l_sorted_nt[i].split(',')[-1]]]
                    N_j = chain[l_sorted_nt[j]][N_dict[l_sorted_nt[j].split(',')[-1]]]
                    dist_N = self.get_euc_dis(N_i, N_j)
                    onehot_N = self.dist_to_onehot(dist_N)
                    dist_N_matrix[i, j, onehot_N] = 1
                except:
                    pass

                if i != j:
                    try:
                        omg = self.get_orientation(C_i, N_i, N_j, C_j)
                        onehot_omg = self.ori_to_onehot(omg, dist_N)
                        ori_omg_matrix[i, j, onehot_omg] = 1
                    except:
                        pass

                    try:
                        theta = self.get_orientation(P_i, C_i, N_i, N_j)
                        onehot_theta = self.ori_to_onehot(theta, dist_N)
                        ori_theta_matrix[i, j, onehot_theta] = 1
                    except:
                        pass

        return dist_P_matrix, dist_C_matrix, dist_N_matrix, ori_omg_matrix, ori_theta_matrix


    def collect_features(self, cif_path, pdbID, chainID, butype):
        try:
            data = self.parse_mmcif(
                path=cif_path,
                file_id=pdbID,
                chain_id=chainID,
                butype=butype,
            )
        except Exception as e:
            print("Oops. {} in parsing {}".format(e, pdbID))

        assert data is not None, "Oops in parsing {}".format(pdbID)
        features_cif = self.processing_features(data)
        d_chain = features_cif.copy()
        l_sorted_nt = self.chain_to_sorted_id(d_chain)
        L = len(l_sorted_nt)
        chi_matrix = np.zeros((L, 24))

        dist_P, dist_C, dist_N, ori_omg, ori_theta = self.get_dist_angle_matrix(d_chain, L, l_sorted_nt)
        eta_bb, theta_bb = self.get_pseudo_torsion_matrix(d_chain, L, l_sorted_nt)

        features_cif = {}
        features_cif["dis_n"] = dist_N
        features_cif["dis_c4"] = dist_C
        features_cif["dis_p"] = dist_P
        features_cif["omg"] = ori_omg
        features_cif["theta"] = ori_theta
        features_cif["eta_bb"] = eta_bb
        features_cif["theta_bb"] = theta_bb
        features_cif["chi"] = chi_matrix

        return features_cif


