import os
from importlib import resources
from turtle import Vec2D
import torch
import sympy
import joblib
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
from gemmi import cif
from tqdm import tqdm
from sympy import sympify
from sympy import Add
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.vectors import Vector
from Bio.PDB.vectors import calc_angle
from Bio.PDB.vectors import rotaxis
from opencomplex.utils.affine_utils import AffineTransformation
opj=os.path.join
from opencomplex.utils.tensor_utils import (
    batched_gather,
)
from opencomplex.np import nucleotide_constants as nc

#################### constant #######################
#####################################################

ang_C3C4C5 = 115
ang_C3C4O4 = 108.1
ang_C2C1O4 = 106.0
# ang_C3C4O4 = 108.1 - 0.061 * tm
bond_length_C3C4 = 1.528
bond_length_C1C2 = 1.525

# need further validation
bond_length_C3O3 = 1.425
bond_length_C2O2 = 1.415

with resources.path("opencomplex.resources", "purine-N1-C1'-C2' or N9-C1'-C2'.joblib") as f:
    gpr_NC1C2_purine = joblib.load(f)
with resources.path("opencomplex.resources", "pyrimidine-N1-C1'-C2' or N9-C1'-C2'.joblib") as f:
    gpr_NC1C2_pyrimidine = joblib.load(f)
with resources.path("opencomplex.resources", "deoxyribose-C2'-endo-C3'-C4'-O4'.joblib") as f:
    gpr_C3C4O4_C2endo = joblib.load(f)

#####################################################

# Load all atom positions from cif file
def get_pos_from_cif(cif_path):
    try:
        # copy all the data from mmCIF file
        doc = cif.read_file(cif_path)
        # mmCIF has exactly one block
        block = doc.sole_block()
        # get target information from block and store in a table
        table = block.find(['_atom_site.group_PDB', '_atom_site.label_asym_id', '_atom_site.label_seq_id', '_atom_site.label_comp_id', '_atom_site.id', '_atom_site.label_atom_id', '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z'])
        # change data format to pandas
        df_pos = pd.DataFrame(table, columns=['group_id', 'chain_id', 'nt_id', 'nt_type', 'atom_id', 'atom_name', 'x', 'y', 'z'])
        # select ATOM type information
        df_pos = df_pos.groupby(['group_id']).get_group('ATOM')
        
        return df_pos

    except Exception as e:
        print("Oops. %s" % e)

# Load target atom position and store in dict
def get_atom_pos(df_pos, chain_id, l_target_atom):
    # input: 
    #        1. dict with ordered all atom pos
    #        2. list of target atom
    # Output: array
    d_atom_pos = {}
    df_pos_chain = df_pos.groupby(['chain_id']).get_group(chain_id)
    for index in df_pos_chain.index:
        atom_name = df_pos_chain.loc[index, 'atom_name']
        if atom_name in l_target_atom:
            nt_id = df_pos_chain.loc[index, 'nt_id']
            nt_type = df_pos_chain.loc[index, 'nt_type']
            nt_symbol = nt_id + '_' + nt_type
            if nt_symbol not in d_atom_pos.keys():
                d_atom_pos[nt_symbol] = {}
            x_, y_, z_ = float(df_pos_chain.loc[index, 'x']), float(df_pos_chain.loc[index, 'y']), float(df_pos_chain.loc[index, 'z'])
            d_atom_pos[nt_symbol][atom_name] = torch.tensor([x_, y_, z_], dtype=torch.float32)
    return d_atom_pos

# Change angle representation from degree to pi
def angle_degree2pi(ang_degree):
    return ang_degree*np.pi/180

# Change angle representation pi degree to degree
def angle_pi2degree(ang_pi):
    return ang_pi*180/np.pi

# Calculate diheral angle
def get_diheral_angle(pos1, pos2, pos3, pos4):
    angle_dihedral = calc_dihedral(Vector(pos1), Vector(pos2), Vector(pos3), Vector(pos4))
    # return dihedral angle in pi format
    return angle_dihedral

# Calculate bond angle
def get_vec_angle(vecA, vecB):
    vecA_norm = get_vec_norm(vecA)
    vecB_norm = get_vec_norm(vecB)
    
    ang_cos = torch.div(torch.sum(torch.dot(vecA, vecB)), vecA_norm * vecB_norm).item()
    ang = angle_pi2degree(np.arccos(ang_cos))
    # return bond angle in degree format
    return ang

# Calculate pseudorotation within the sugar ring
# The output Tm angle is in degree format
def calculate_pseudorotation(TAU_0, TAU_1, TAU_2, TAU_3, TAU_4):
    _theta = [TAU_2, TAU_3, TAU_4, TAU_0, TAU_1]

    sum_sin = 0.0
    sum_cos = 0.0

    for i_t, t in enumerate(_theta):
        x = 0.8 * math.pi * i_t
        sum_sin += t * math.sin(x)
        sum_cos += t * math.cos(x)

    P_deg = math.degrees(math.atan2(-sum_sin, sum_cos))

    if P_deg < 0.0:
        P_deg += 360.0

    P_rad = math.radians(P_deg)
    Tm = 0.4 * (math.cos(P_rad) * sum_cos - math.sin(P_rad) * sum_sin)

    return Tm

# Calculate L2 normalization
def get_vec_norm(vec):
    # return torch.norm(vec.detach().numpy(), p=2, dim=0).item()
    return np.linalg.norm(vec.detach().cpu().numpy(), ord=2)

# Get dot result of two vectors according to the value of cos(theta)
# The input angle is in pi format
def get_dot_from_angle(angle, norm_A, norm_B):
    cos_ = np.cos(angle_degree2pi(angle))
    return float(cos_ * norm_A * norm_B)

# Get normal vector
def get_norm_vec(vecA, vecB):
    # get the normal vector for vector ab and ac
    # In which vecA represent ab, vecB represent ac
    return(torch.tensor([vecA[1] * vecB[2] - vecA[2] * vecB[1],
                         vecA[2] * vecB[0] - vecA[0] * vecB[2],
                         vecA[0] * vecB[1] - vecA[1] * vecB[0]
                         ], dtype=torch.float32))

# GPR model prediction
def gpr(gpr_model, chi):
    return gpr_model.predict(np.array([chi]).reshape(-1, 1))[0]

# Predict atom position
def get_pred_bb_pos(pos_center, pos_upperLeft, pos_upperRight, ang_left, ang_right, bond_length):
    
    if pos_upperLeft.device != pos_center.device or pos_upperRight.device != pos_center.device:
        print('device not match', pos_center.device, pos_upperLeft.device, pos_upperRight.device)
    
    vec_left = pos_upperLeft - pos_center
    vec_right = pos_upperRight - pos_center
    
    if torch.count_nonzero(vec_left).item() == 0 or torch.count_nonzero(vec_right).item() == 0:
        return torch.zeros((1, 3) , dtype=torch.float32, device=pos_center.device)
    
    x = sympy.Symbol('x', real=True)
    y = sympy.Symbol('y', real=True)
    z = sympy.Symbol('z', real=True)
    
    cons1 = get_dot_from_angle(ang_left, get_vec_norm(vec_left), bond_length)
    cons2 = get_dot_from_angle(ang_right, get_vec_norm(vec_right), bond_length)
    
    vec_left_np = vec_left.detach().cpu().numpy()
    vec_right_np = vec_left.detach().cpu().numpy()
    
    # alg1 = sympify(vec_left[0]*x + vec_left[1]*y + vec_left[2]*z - cons1)
    # alg2 = sympify(vec_right[0]*x + vec_right[1]*y + vec_right[2]*z - cons2)
    alg1 = sympify(vec_left_np[0]*x + vec_left_np[1]*y + vec_left_np[2]*z - cons1)
    alg2 = sympify(vec_right_np[0]*x + vec_right_np[1]*y + vec_right_np[2]*z - cons2)
    alg3 = sympify(x*x + y*y + z*z - bond_length*bond_length)
    
    solved_value=sympy.solve(
        [alg1, alg2, alg3], 
        [x, y, z]
        )
    # which means that the solved coord is in the complex domain
    if len(solved_value) == 0:
        return torch.zeros((1, 3) , dtype=torch.float32, device=pos_center.device)
    
    # print(alg1, alg2, alg3, solved_value)
    
    candidate_vec_1 = torch.tensor(solved_value[0], dtype=torch.float32)
    candidate_vec_2 = torch.tensor(solved_value[1], dtype=torch.float32)
    
    norm_vec = get_norm_vec(vec_left, vec_right)
    
    ang1 = get_vec_angle(norm_vec, candidate_vec_1)
    ang2 = get_vec_angle(norm_vec, candidate_vec_2)
    
    # Need more validation
    if ang1 > ang2:
        return torch.tensor([solved_value[0][0] + pos_center[0], solved_value[0][1] + pos_center[1], solved_value[0][2] + pos_center[2]], dtype=torch.float32, device=pos_center.device)
    else:
        return torch.tensor([solved_value[1][0] + pos_center[0], solved_value[1][1] + pos_center[1], solved_value[1][2] + pos_center[2]], dtype=torch.float32, device=pos_center.device)
    
################# pred side chain ####################
#####################################################

def centroid(X):
    C = sum(X)/len(X)
    return C

def kabsch(P, Q):
    C = torch.mm(torch.transpose(P, 0, 1), Q)
    V, S, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:,-1]
    U = torch.mm(V,W)
    return U

def rotate(P, Q):
    U = kabsch(P, Q)
    P = torch.mm(P, U)
    return P

# coordinate transform function
def coord_trans(trans_system, atomType):

    if atomType == 'O5':
        origin_system = torch.tensor([
            [48.008,  17.712, -11.737],
            [47.776,  18.174, -10.309],
            [46.580,  17.585,  -9.594]
            ], dtype=torch.float32)
        trans_object = torch.tensor([[47.008,  16.769, -12.147]], dtype=torch.float32)
        
    if atomType == 'P':
        origin_system = torch.tensor([
            [47.008,  16.769, -12.147],
            [48.008,  17.712, -11.737],
            [48.694,  18.343, -12.936]
            ], dtype=torch.float32)
        trans_object = torch.tensor([[46.729,  16.490, -13.697]], dtype=torch.float32)
        
    if atomType == 'C':
        origin_system = torch.tensor([
            [44.641,  18.117,  -9.192],
            [45.942,  18.724,  -8.956],
            [46.556,  18.946, -10.215]
            ], dtype=torch.float32)
        trans_object = torch.tensor([[44.224,  16.881,  -8.774]], dtype=torch.float32)
    
    #translation normalization
    origin_translation = centroid(origin_system)
    origin_system_cen = origin_system - origin_translation
    
    trans_translation = centroid(trans_system)
    trans_system_cen = trans_system - trans_translation
    
    #get rotation matrix
    rotate_matrix = kabsch(origin_system_cen, trans_system_cen)
    
    trans_object_cen = trans_object - origin_translation
    
    result = torch.mm(trans_object_cen, rotate_matrix) + trans_translation
    
    return np.around(result[0], 3)


def from_torsion_to_coodinate(pos1, pos2, pos3, torsion, atomType):
    # pos1 refers to the position of atom that close to predicted atom
    # pos3 refers to the position of atom that far from predicted atom
    # torsion is defined as the diheral angle between pred_pos, pos1, pos2, pos3
    three_atom_pos = torch.stack([pos1, pos2, pos3], dim=0)
    
    init_pred_pos = coord_trans(three_atom_pos, atomType)
    init_vector = init_pred_pos - pos2
    axis_vector = pos1 - pos2
    
    rot_matrix = rotaxis(torsion, Vector(axis_vector[0], axis_vector[1], axis_vector[2]))
    end_vector = Vector(init_vector[0], init_vector[1], init_vector[2]).left_multiply(rot_matrix)
    
    final_pred_pos = pos2 + torch.tensor(end_vector.get_array(), dtype=torch.float32)
    
    return final_pred_pos


#################### pred base ######################
#####################################################
#input
#N9,C4

#output
#C2,C4,C5,C6,C8,N1,N2,N3,N7,N9,O6
def get_base_coord_G(trans_system):
    
    origin_system = torch.tensor([[-69.997,18.097,3.605],
                             [-71.051,18.750,3.009]], dtype=torch.float32, device=trans_system.device)
    
    trans_object = torch.tensor([[-72.544,20.323,2.577],
                             [-71.051,18.750,3.009],
                             [-71.508,17.888,2.032],
                             [-72.608,18.276,1.225],
                             [-69.880,16.888,2.962],
                             [-73.068,19.541,1.578],
                             [-73.129,21.516,2.759],
                             [-71.517,19.972,3.335],
                             [-70.761,16.720,2.014],
                             [-69.997,18.097,3.605],
                             [-73.154,17.646,0.309]], dtype=torch.float32, device=trans_system.device)
    
    #translation normalization
    origin_translation = centroid(origin_system)
    origin_system_cen = origin_system - origin_translation
    
    trans_translation = centroid(trans_system)
    trans_system_cen = trans_system - trans_translation
    
    #get rotation matrix
    rotate_matrix = kabsch(origin_system_cen, trans_system_cen)
    
    trans_object_cen = trans_object - origin_translation
    result = torch.mm(trans_object_cen, rotate_matrix) + trans_translation
    
    return result.to(trans_system.device)
    return np.around(result, 3)


#input
#N9,C4

#output
#C2,C4,C5,C6,C8,N1,N3,N6,N7,N9
def get_base_coord_A(trans_system):
    
    origin_system = torch.tensor([[-85.057,11.184,4.979],
                             [-84.068,10.387,4.462]], dtype=torch.float32, device=trans_system.device)
    
    trans_object = torch.tensor([[-82.995,8.955,3.173],
                             [-84.068,10.387,4.462],
                             [-82.961,10.597,5.265],
                             [-81.792,9.887,4.924],
                             [-84.508,11.818,6.064],
                             [-81.846,9.057,3.852],
                             [-84.153,9.575,3.392],
                             [-80.646,9.979,5.606],
                             [-83.254,11.500,6.278],
                             [-85.057,11.184,4.979]], dtype=torch.float32, device=trans_system.device)
    
    #translation normalization
    origin_translation = centroid(origin_system)
    origin_system_cen = origin_system - origin_translation
    
    trans_translation = centroid(trans_system)
    trans_system_cen = trans_system - trans_translation
    
    #get rotation matrix
    rotate_matrix = kabsch(origin_system_cen,trans_system_cen)
    
    trans_object_cen = trans_object - origin_translation
    result = torch.mm(trans_object_cen, rotate_matrix) + trans_translation
    return result.to(trans_system.device)
    return np.around(result,3)

#input
#N1,C2

#output
#C2,C4,C5,C6,N1,N3,O2,O4
def get_base_coord_U_C(trans_system):
    
    origin_system = torch.tensor([[-83.152,15.276,5.512],
                             [-83.009,14.396,4.453]], dtype=torch.float32, device=trans_system.device)
    
    trans_object = torch.tensor([[-80.856,13.681,5.420],
                             [-83.009,14.396,4.453],
                             [-81.082,14.617,6.484],
                             [-82.196,15.361,6.496],
                             [-83.152,15.276,5.512],
                             [-81.858,13.645,4.468],
                             [-83.837,14.287,3.565],
                             [-79.874,12.941,5.297]
                             ], dtype=torch.float32, device=trans_system.device)
    
    #translation normalization
    origin_translation = centroid(origin_system)
    origin_system_cen = origin_system - origin_translation
    
    trans_translation = centroid(trans_system)
    trans_system_cen = trans_system - trans_translation
    
    #get rotation matrix
    rotate_matrix = kabsch(origin_system_cen,trans_system_cen)
    
    trans_object_cen = trans_object - origin_translation
    result = torch.mm(trans_object_cen,rotate_matrix) + trans_translation
    
    return result.to(trans_system.device)
    return np.around(result,3)


def get_base_coord(N_pos, C_base_pos, ntType):
    # A
    if ntType == 'A':
        #input: N9,C4
        #output order: C2,C4,C5,C6,C8,N1,N3,N6,N7,N9
        base_atom_pos_part = get_base_coord_A(torch.stack([N_pos, C_base_pos], dim=0))
        base_atom_pos = torch.cat([
            base_atom_pos_part[5:6, :],
            base_atom_pos_part.new_zeros((1, 3)),
            base_atom_pos_part[6:7, :],
            base_atom_pos_part.new_zeros((1, 3)),
            base_atom_pos_part[7:10, :],
            base_atom_pos_part[0:5, :],
            base_atom_pos_part.new_zeros((3, 3)),
        ], dim=0
        )
    # G
    elif ntType == 'G':
        #input: N9,C4
        #output order: C2,C4,C5,C6,C8,N1,N2,N3,N7,N9,O6
        base_atom_pos_part = get_base_coord_G(torch.stack([N_pos, C_base_pos], dim=0))
        base_atom_pos = torch.cat([
            base_atom_pos_part[5:8, :],
            base_atom_pos_part.new_zeros((2, 3)),
            base_atom_pos_part[8:10, :],
            base_atom_pos_part[0:5, :],
            base_atom_pos_part.new_zeros((2, 3)),
            base_atom_pos_part[10:11, :],
        ], dim=0
        )
    # U or C
    elif ntType == 'U' or ntType == 'C':
        #input: N1,C2
        #output order: C2,C4,C5,C6,N1,N3,O2,O4
        base_atom_pos_part = get_base_coord_U_C(torch.stack([N_pos, C_base_pos], dim=0))
        base_atom_pos = torch.cat([
            base_atom_pos_part[4:5, :],
            base_atom_pos_part.new_zeros((1, 3)),
            base_atom_pos_part[5:6, :],
            base_atom_pos_part.new_zeros((4, 3)),
            base_atom_pos_part[0:4, :],
            base_atom_pos_part.new_zeros((1, 3)),
            base_atom_pos_part[6:8, :],
            base_atom_pos_part.new_zeros((1, 3)),
        ], dim=0
        )
    else:
        # print(ntType, len(ntType), type(ntType))
        base_atom_pos = N_pos.new_zeros((15, 3))
    
    return base_atom_pos

#####################################################

def torsion_angles_to_frames(
    alpha: torch.Tensor,
    butype: torch.Tensor,
):
    B, N = alpha.shape[:2]
    bb_rot = alpha.new_zeros((B, N, 5, 2))
    bb_rot[..., 1] = 1

    # [*, N, 9, 2]
    alpha = torch.cat([bb_rot, alpha], dim=-2)

    # [*, N, 9, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros((*alpha.shape[:-1], 3, 3))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_frames = AffineTransformation(all_rots, None)

    gamma_frame_to_frame = all_frames[..., 5]
    beta_frame_to_frame = all_frames[..., 6]
    alpha_frame_to_frame = all_frames[..., 7]
    chi_frame_to_frame = all_frames[..., 8]

    gamma_frame_to_bb = gamma_frame_to_frame
    beta_frame_to_bb = gamma_frame_to_bb*beta_frame_to_frame
    alpha_frame_to_bb = beta_frame_to_bb*alpha_frame_to_frame
    chi_frame_to_bb = chi_frame_to_frame

    all_frames_to_bb = AffineTransformation.cat(
        [ 
            all_frames[..., :5],
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha_frame_to_bb.unsqueeze(-1),
            chi_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    return all_frames_to_bb


def frames_and_literature_positions_to_atom14_pos(
    r,
    butype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # # [*, N, 14, 4, 4]
    # default_4x4 = default_frames[butype, ...]

    # # [*, N, 14]
    # group_mask = group_idx[butype, ...]

    # # [*, N, 14, 8]
    # group_mask = nn.functional.one_hot(
    #     group_mask,
    #     num_classes=default_frames.shape[-3],
    # )

    # # [*, N, 14, 8]
    # t_atoms_to_global = r[..., None, :] * group_mask

    # # [*, N, 14]
    # t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
    #     lambda x: torch.sum(x, dim=-1)
    # )

    # # [*, N, 14, 1]
    # atom_mask = atom_mask[butype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[butype, ...]
    pred_positions = AffineTransformation.apply_affine(r, lit_positions)
    # pred_positions = pred_positions * atom_mask

    return pred_positions

###########################################################

def single_NT_5_atom_pos(C5_pos, C4_pos, O4_pos, C1_pos, N_pos, alpha, beta, gamma, delta, chi, ntType):
    # Input: C5', C4', O4', C1', N_base (N1/N9), alpha, beta, gamma, delta, v1, chi, ntype
    #        position dim is 1，3
    #        torsion dim is 1
    #        angle dim is 1
    # Output: C3', C2', O3', O2', C_base (C2/C4)
    
    if abs(angle_pi2degree(chi.item())) > 90:
        ang_C4C3O3 = 110.7
        # ang_C2C3O3 = 109.4
        ang_C2C3O3 = 110.7
        ang_C3C2O2 = 113.6
        ang_C1C2O2 = 112.0
    # x syn
    else:
        ang_C4C3O3 = 109.8
        # ang_C2C3O3 = 110.1
        ang_C2C3O3 = 110.8
        ang_C3C2O2 = 114.1
        ang_C1C2O2 = 112.5
    
    if int(ntType) in [1, 3]:
        ang_NC1C2 = gpr(gpr_NC1C2_purine, angle_pi2degree(chi.item()))
    elif int(ntType) in [2, 4]:
        ang_NC1C2 = gpr(gpr_NC1C2_pyrimidine, angle_pi2degree(chi.item()))
    else:
        ang_NC1C2 = (gpr(gpr_NC1C2_purine, angle_pi2degree(chi.item())) + gpr(gpr_NC1C2_pyrimidine, angle_pi2degree(chi.item()))) / 2
    
    C3_pos_cons = get_pred_bb_pos(C4_pos, C5_pos, O4_pos, ang_C3C4C5, ang_C3C4O4, bond_length_C3C4)
    C2_pos_cons = get_pred_bb_pos(C1_pos, O4_pos, N_pos, ang_C2C1O4, ang_NC1C2, bond_length_C1C2)
    O3_pos_cons = get_pred_bb_pos(C3_pos_cons, C4_pos, C2_pos_cons, ang_C4C3O3, ang_C2C3O3, bond_length_C3O3)
    O2_pos_cons = get_pred_bb_pos(C2_pos_cons, C3_pos_cons, C1_pos, ang_C3C2O2, ang_C1C2O2, bond_length_C2O2)
    C_base_pose_cons = from_torsion_to_coodinate(N_pos, C1_pos, O4_pos, angle_pi2degree(chi.item()), 'C')
    
    return torch.stack([C3_pos_cons, C2_pos_cons, O3_pos_cons, O2_pos_cons, C_base_pose_cons], dim=0)


# def single_NT_all_atom_pos(C5_pos, C4_pos, O4_pos, C1_pos, N_pos, alpha, beta, gamma, delta, chi, ntType):
#     # Input: C5', C4', O4', C1', N_base (N1/N9), alpha, beta, gamma, delta, v1, chi, ntype
#     # Output: dict of all-atom positions
    
#     d_all_atom_pos = {}
    
#     five_atom_pos = frames_to_5_atom_pos(C5_pos, C4_pos, O4_pos, C1_pos, N_pos, alpha, beta, gamma, delta, chi, ntType)
#     C3_pos_cons = five_atom_pos[0]
#     C2_pos_cons = five_atom_pos[1]
#     O3_pos_cons = five_atom_pos[2]
#     O2_pos_cons = five_atom_pos[3]
#     C_base_pose_cons = five_atom_pos[4]
    
#     O5_pos_cons= from_torsion_to_coodinate(C5_pos, C4_pos, C3_pos_cons, gamma, 'O5')
#     P_pos_cons = from_torsion_to_coodinate(O5_pos_cons, C5_pos, C4_pos, beta, 'P')
#     # A
#     if int(ntType) == 1:
#         #input: N9,C4
#         #output order: C2,C5,C6,C8,N1,N3,N6,N7
#         base_atom_pos = get_base_coord_A(torch.stack([N_pos, C_base_pose_cons], dim=0))
#     # G
#     if int(ntType) == 3:
#         #input: N9,C4
#         #output order: C2,C4,C5,C6,C8,N1,N2,N3,N7,N9,O6
#         base_atom_pos = get_base_coord_G(torch.stack([N_pos, C_base_pose_cons], dim=0))
#     # U or C
#     if int(ntType) == 2 or int(ntType) == 4:
#         #input: N1,C2
#         #output order: C4,C5,C6,N3,O2,O4
#         base_atom_pos = get_base_coord_U_C(torch.stack([N_pos, C_base_pose_cons], dim=0))
        
#     # In the order of C1' C2' C3' C4' O4' O3' O2'
#     d_all_atom_pos['sugar'] = torch.stack([C1_pos, C2_pos_cons, C3_pos_cons, C4_pos, O4_pos, O3_pos_cons, O2_pos_cons], dim=0)
#     # In the order of C5' O5' P
#     d_all_atom_pos['side_chain'] = torch.stack([C5_pos, O5_pos_cons, P_pos_cons], dim=0)
#     d_all_atom_pos['base'] = base_atom_pos
    
#     return d_all_atom_pos


def single_NT_all_atom_pos(atom9_positions, torsions, ntType):
    # Input: atom9_positions (9 * 3), torsion (4 * 2), ntype (A, U, G, C)
    #        The 9 atom is organized as C5', C4', O4', C1', N, O5', P, O3', C base.
    # Output: all-atom positions for single NT (28 * 3)
    device = atom9_positions.device

    C5_pos = atom9_positions[0].to(device)
    C4_pos = atom9_positions[1].to(device)
    O4_pos = atom9_positions[2].to(device)
    C1_pos = atom9_positions[3].to(device)
    N_pos = atom9_positions[4].to(device)
    C_base_pos = atom9_positions[-1].to(device)
    
    # print(C4_pos.device, C5_pos.device, O4_pos.device)
    
    chi_sincos = torsions[3]
    # chi_sincos.shape
    # print(chi_sincos[-1])
    # chi_pi = math.acos(chi_sincos[-1].item())
    chi_pi = np.arccos(chi_sincos[-1].item())
    # print(torsions.shape)
    # print(torsions[3].shape)
    # print(torsions[3])
    # print(chi_sincos)
    # print(chi_sincos[-1].item())
    # print(chi_pi)
    
    if abs(angle_pi2degree(chi_pi)) > 90:
        ang_C3C2O2 = 113.6
        ang_C1C2O2 = 112.0
    # x syn
    else:
        ang_C3C2O2 = 114.1
        ang_C1C2O2 = 112.5
        
    if ntType in ['A', 'G']:
        ang_NC1C2 = gpr(gpr_NC1C2_purine, angle_pi2degree(chi_pi))
    elif ntType in ['C', 'U']:
        ang_NC1C2 = gpr(gpr_NC1C2_pyrimidine, angle_pi2degree(chi_pi))
    else:
        ang_NC1C2 = (gpr(gpr_NC1C2_purine, angle_pi2degree(chi_pi)) + gpr(gpr_NC1C2_pyrimidine, angle_pi2degree(chi_pi))) / 2
    
    C3_pos_cons = get_pred_bb_pos(C4_pos, C5_pos, O4_pos, ang_C3C4C5, ang_C3C4O4, bond_length_C3C4)
    C2_pos_cons = get_pred_bb_pos(C1_pos, O4_pos, N_pos, ang_C2C1O4, ang_NC1C2, bond_length_C1C2)
    # O3_pos_cons = get_pred_bb_pos(C3_pos_cons, C4_pos, C2_pos_cons, ang_C4C3O3, ang_C2C3O3, bond_length_C3O3)
    # print('cons', C3_pos_cons.device, C2_pos_cons.device)
    O2_pos_cons = get_pred_bb_pos(C2_pos_cons, C3_pos_cons, C1_pos, ang_C3C2O2, ang_C1C2O2, bond_length_C2O2)
    
    base_pos_cons = get_base_coord(N_pos, C_base_pos, ntType)
    
    # print(C2_pos_cons, C3_pos_cons, O2_pos_cons)
    
    # print(C2_pos_cons.reshape(1, 3).shape, atom9_positions[3:4, :].shape, base_pos_cons.shape)
    
    all_atom_pos_NT = torch.cat(
        [
            atom9_positions[3:4, :],
            C2_pos_cons.reshape(1, 3), 
            C3_pos_cons.reshape(1, 3),
            atom9_positions[1:2, :],
            atom9_positions[0:1, :],
            atom9_positions[5:6, :],
            atom9_positions[2:3, :],
            atom9_positions[7:8, :],
            O2_pos_cons.reshape(1, 3),
            atom9_positions[6:7, :],
            atom9_positions.new_zeros((3, 3)),
            base_pos_cons,
        ], dim=0,
    )
    
    # print(all_atom_pos_NT.shape)
    
    return all_atom_pos_NT.to(device)


def frames_to_5_atom_pos(chain_C5_pos, chain_C4_pos, chain_O4_pos, chain_C1_pos, chain_N_pos, chain_alpha, chain_beta, chain_gamma, chain_delta, chain_chi, chain_ntType):
    # Input: C5', C4', O4', C1', N_base (N1/N9), alpha, beta, gamma, delta, v1, chi, ntype
    #        position dim is B，N，3
    #        torsion dim is B，N，2
    #        angle dim is B，N
    # Output: C3', C2', O3', O2', C_base (C2/C4)
    l_nt_pos = []
    for chain_index in range(chain_C5_pos.shape[0]):
        for NT_index in range(chain_C5_pos.shape[1]):
            
            single_NT_pos = single_NT_5_atom_pos(chain_C5_pos[chain_index][NT_index],
                                 chain_C4_pos[chain_index][NT_index],
                                 chain_O4_pos[chain_index][NT_index],
                                 chain_C1_pos[chain_index][NT_index],
                                 chain_N_pos[chain_index][NT_index],
                                 chain_alpha[chain_index][NT_index],
                                 chain_beta[chain_index][NT_index],
                                 chain_gamma[chain_index][NT_index],
                                 chain_delta[chain_index][NT_index],
                                 chain_chi[chain_index][NT_index],
                                 chain_ntType[chain_index][NT_index]
                                 )
            
            l_nt_pos.append(single_NT_pos)
    
    return torch.unsqueeze(torch.stack(l_nt_pos, dim=0), dim=0)


# get sparse representation of 9 atoms
def atom9_to_atom28_train(atom9, batch):
    atom28_data = batched_gather(
        atom9,
        batch["residx_atom28_to_atom9"],
        dim=-2,
        no_batch_dims=len(atom9.shape[:-2]),
    )

    atom28_data = atom28_data * batch["atom28_atom_exists"][..., None]

    return atom28_data


def atom9_to_atom28_infer(sm, batch):
    
    all_atom_pos_pred = []
    
    butype = batch['butype'][0]
    # N, 9, 3. Remove batch dims
    atom9_positions = sm['positions'][-1][0]
    # N, 4, 2. Remove batch dims
    angles = sm['angles'][-1][0]
    
    for nt_idx in range(butype.shape[0]):
        # butype_idx = butype[nt_idx].item()
        butype_idx = butype[nt_idx][0].item()
        # A, U, G, C
        # ntname = dict(zip(nc.butype_order.values(), nc.butype_order.keys()))[butype_idx]
        ntname = dict(zip(nc.butype_order_with_x.values(), nc.butype_order_with_x.keys()))[butype_idx]
        # print(nt_idx, ntname)
        all_atom_pos_pred.append(single_NT_all_atom_pos(atom9_positions[nt_idx], angles[nt_idx], ntname))
    
    atom28_data = torch.stack(all_atom_pos_pred, dim=0).unsqueeze(dim=0)
    
    # atom28_data = atom28_data * batch["atom28_atom_exists"][..., None]
    ## remove the recycling dim
    atom28_data = atom28_data * batch["atom28_atom_exists"][..., 0:1]
    
    return atom28_data