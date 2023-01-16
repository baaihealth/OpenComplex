"""Constants used in RNAFold."""

import collections
import functools
from typing import Mapping, List, Tuple
from importlib import resources

import numpy as np
import tree


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "U",
    "G",
    "C"
]

restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 4.
# The index of unknown NT type is 4
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

# Not sure whether there will be X in RNA sequence
restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(
    sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An RNA sequence.
      mapping: A dictionary mapping nucleotides to integers.
      map_unknown_to_x: If True, any nucleotide that is not in the mapping will be
        mapped to the unknown nucleotide 'X'. If the mapping doesn't contain
        nucleotide 'X', an error will be thrown. If False, any nucleotide not in
        the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_nts) with one-hot encoding of
      the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_nts - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for nt_index, nt_type in enumerate(sequence):
        if map_unknown_to_x:
            if nt_type.isalpha() and nt_type.isupper():
                nt_id = mapping.get(nt_type, mapping["X"])
            else:
                raise ValueError(
                    f"Invalid character in the sequence: {nt_type}"
                )
        else:
            nt_id = mapping[nt_type]
        one_hot_arr[nt_index, nt_id] = 1

    return one_hot_arr

# Atoms positions relative to the rigid groups, defined by the alpha, beta, revised gamma, and chi.
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
#
# The statistical results are as follows:
# from C5' C4' and O4' to O5', according to torsion revised gamma (the typical gamma is calculated according to the positions of O5' C5' C4' and C3')
# G  mean xyz:  2.010, 1.328, 0.000   var xyz:  0.002 0.001 0.000
# C  mean xyz:  2.009, 1.329, 0.000   var xyz:  0.002 0.001 0.000
# U  mean xyz:  2.010, 1.330, 0.000   var xyz:  0.003 0.000 0.000
# A  mean xyz:  2.008, 1.330, 0.000   var xyz:  0.002 0.000 0.000
#
# from O5' C5' and C4' to P, according to torsion beta
# G  mean xyz:  2.237, 1.370, 0.000   var xyz:  0.002 0.001 0.000
# C  mean xyz:  2.234, 1.371, 0.000   var xyz:  0.003 0.001 0.000
# U  mean xyz:  2.233, 1.373, 0.000   var xyz:  0.002 0.001 0.000
# A  mean xyz:  2.232, 1.372, 0.000   var xyz:  0.002 0.000 0.000
#
# from P O5' and C5' to O3' in previous NT, according to torsion alpha
# G  mean xyz:  1.888, 2.560, 0.000   var xyz:  0.218 0.020 0.000
# C  mean xyz:  1.888, 2.561, 0.000   var xyz:  0.188 0.015 0.000
# U  mean xyz:  2.020, 2.531, 0.000   var xyz:  0.276 0.033 0.000
# A  mean xyz:  2.072, 2.545, 0.000   var xyz:  0.337 0.256 0.000
#
# from N C1' and O4' to C4/C2, according to torsion chi
# G  mean xyz:  2.278, 1.107, 0.000   var xyz:  0.002 0.000 0.000
# C  mean xyz:  4.176, 0.132, 0.000   var xyz:  0.001 0.005 0.000
# U  mean xyz:  4.252, 0.109, 0.000   var xyz:  0.001 0.005 0.000
# A  mean xyz:  2.285, 1.104, 0.000   var xyz:  0.001 0.001 0.000
# for C5'
# G  mean xyz:  1.510, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# C  mean xyz:  1.509, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# U  mean xyz:  1.511, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# A  mean xyz:  1.508, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# for C4'
# G  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# C  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# U  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# A  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# for O4'
# G  mean xyz:  -0.487, 1.369, 0.000   var xyz:  0.002 0.0 0.0
# C  mean xyz:  -0.487, 1.367, 0.000   var xyz:  0.001 0.0 0.0
# U  mean xyz:  -0.485, 1.367, 0.000   var xyz:  0.002 0.0 0.0
# A  mean xyz:  -0.483, 1.368, 0.000   var xyz:  0.001 0.0 0.0
# for C1'
# G  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# C  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# U  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# A  mean xyz:  0.000, 0.000, 0.000   var xyz:  0.000 0.000 0.000
# for N
# G  mean xyz:  1.463, 0.000, 0.000   var xyz:  0.0 0.0 0.0
# C  mean xyz:  1.476, 0.000, 0.000   var xyz:  0.0 0.0 0.0
# U  mean xyz:  1.473, 0.000, 0.000   var xyz:  0.0 0.0 0.0
# A  mean xyz:  1.466, 0.000, 0.000   var xyz:  0.0 0.0 0.0

# rigid_group_atom_positions = {
#     'A': [
#         ["C5'", 0, (1.508, 0.000, 0.000)],
#         ["C4'", 0, (0.000, 0.000, 0.000)],
#         ["O4'", 0, (-0.483, 1.368, 0.000)],
#         ["N9", 0, (1.466, 0.000, 0.000)],
#         ["C1'", 0, (0.000, 0.000, 0.000)],
#         ["O5'", 0, (2.008, 1.330, 0.000)],
#         ["P", 0, (2.232, 1.372, 0.000 )],
#         ["O3'", 0, (2.072, 2.545, 0.000)],
#         ["C4", 0, (2.285, 1.104, 0.000)],
#     ],
#     'U': [
#         ["C5'", 0, (1.511, 0.000, 0.000)],
#         ["C4'", 0, (0.000, 0.000, 0.000)],
#         ["O4'", 0, (-0.485, 1.367, 0.000)],
#         ["N1", 0, (1.473, 0.000, 0.000)],
#         ["C1'", 0, (0.000, 0.000, 0.000)],
#         ["O5'", 0, (2.010, 1.330, 0.000)],
#         ['P', 0, (2.233, 1.373, 0.000)],
#         ["O3'", 0, (2.020, 2.531, 0.000)],
#         ["C2", 0, (4.252, 0.109, 0.000)],
#     ],
#     'G': [
#         ["C5'", 0, (1.510, 0.000, 0.000)],
#         ["C4'", 0, (0.000, 0.000, 0.000)],
#         ["O4'", 0, (-0.487, 1.369, 0.000)],
#         ["N9", 0, (1.463, 0.000, 0.000)],
#         ["C1'", 0, (0.000, 0.000, 0.000)],
#         ["O5'", 0, (2.010, 1.328, 0.000)],
#         ["P", 0, (2.237, 1.370, 0.000)],
#         ["O3'", 0, (1.888, 2.560, 0.000)],
#         ["C4", 0, (2.278, 1.107, 0.000)],
#     ],
#     'C': [
#         ["C5'", 0, (1.509, 0.000, 0.000)],
#         ["C4'", 0, (0.000, 0.000, 0.000)],
#         ["O4'", 0, (-0.487, 1.367, 0.000)],
#         ["N1", 0, (1.476, 0.000, 0.000)],
#         ["C1'", 0, (0.000, 0.000, 0.000)],
#         # ["O5'", 0, (2.009, 1.329, 0.000)],
#         ["O5'", 0, (2.006, 1.33, 0.000)],
#         ["P", 0, (1.888, 2.561, 0.000)],
#         ["O3'", 0, (1.888, 2.561, 0.000)],
#         ["C2", 0, (4.176, 0.132, 0.000)],
#     ],
# }


# ## C5'
# A mean xyz:  1.508, 0.0, 0.0     var xyz:  0.0 0.0 0.0
# U mean xyz:  1.511, 0.0, 0.0     var xyz:  0.0 0.0 0.0
# G mean xyz:  1.510, 0.0, 0.0     var xyz:  0.0 0.0 0.0
# C mean xyz:  1.509, 0.0, 0.0     var xyz:  0.0 0.0 0.0
# ## C4'
# A mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# U mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# G mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# C mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# ## O4'
# A mean xyz:  -0.483, 1.368, 0.0     var xyz:  0.001 0.0 0.0
# U mean xyz:  -0.485, 1.367, 0.0     var xyz:  0.002 0.0 0.0
# G mean xyz:  -0.487, 1.369, 0.0     var xyz:  0.002 0.0 0.0
# C mean xyz:  -0.487, 1.367, 0.0     var xyz:  0.001 0.0 0.0
# ## N base
# A mean xyz:  2.537, 0.0, 0.0     var xyz:  0.001 0.0 0.0
# U mean xyz:  2.440, 0.0, 0.0     var xyz:  0.001 0.0 0.0
# G mean xyz:  2.533, 0.0, 0.0     var xyz:  0.001 0.0 0.0
# C mean xyz:  2.467, 0.0, 0.0     var xyz:  0.001 0.0 0.0
# ## C1'
# A mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# U mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# G mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# C mean xyz:  0.0 0.0 0.0     var xyz:  0.0 0.0 0.0
# ## O5'
# A mean xyz:  2.008, 1.330, 0.0     var xyz:  0.002 0.0 0.0
# U mean xyz:  2.010, 1.329, 0.0     var xyz:  0.003 0.0 0.0
# G mean xyz:  2.010, 1.328, 0.0     var xyz:  0.002 0.001 0.0
# C mean xyz:  2.009, 1.329, 0.0     var xyz:  0.002 0.001 0.0
# ## P
# A mean xyz:  2.232, 1.372, 0.0     var xyz:  0.002 0.0 0.0
# U mean xyz:  2.404, 1.482, 0.0     var xyz:  2.043 0.833 0.0
# G mean xyz:  2.237, 1.370, 0.0     var xyz:  0.002 0.001 0.0
# C mean xyz:  2.171, 1.598, 0.0     var xyz:  2.456 2.535 0.0
# ## O3'
# A mean xyz:  2.019, 1.625, 0.0     var xyz:  0.27 0.527 0.0
# U mean xyz:  1.983, 1.558, 0.0     var xyz:  0.004 0.0 0.0
# G mean xyz:  1.997, 2.078, 0.0     var xyz:  3.697 9.504 0.0
# C mean xyz:  1.968, 1.560, 0.0     var xyz:  0.002 0.0 0.0
# ## C base
# A mean xyz:  1.320, 0.638, 0.0     var xyz:  0.0 0.0 0.0
# U mean xyz:  1.275, 0.738, 0.0     var xyz:  0.0 0.0 0.0
# G mean xyz:  1.316, 0.639, 0.0     var xyz:  0.0 0.0 0.0
# C mean xyz:  1.281, 0.733, 0.0     var xyz:  0.0 0.0 0.0

rigid_group_atom_positions = {
    'A': [
        ["C5'", 0, (1.508, 0.000, 0.000)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (-0.483, 1.368, 0.000)],
        ["N9", 0, (2.537, 0.000, 0.000)],
        ["C1'", 0, (0.000, 0.000, 0.000)],
        ["O5'", 0, (2.008, 1.330, 0.000)],
        ["P", 0, (2.232, 1.372, 0.000 )],
        ["O3'", 0, (2.019, 1.625, 0.000)],
        ["C4", 0, (1.320, 0.638, 0.000)],
    ],
    'U': [
        ["C5'", 0, (1.511, 0.0, 0.0)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (-0.485, 1.367, 0.0)],
        ["N1", 0, (2.440, 0.000, 0.000)],
        ["C1'", 0, (0.000, 0.000, 0.000)],
        ["O5'", 0, (2.010, 1.329, 0.000)],
        ['P', 0, (2.404, 1.482, 0.000)],
        ["O3'", 0, (1.983, 1.558, 0.000)],
        ["C2", 0, (1.275, 0.738, 0.000)],
    ],
    'G': [
        ["C5'", 0, (1.510, 0.000, 0.000)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (-0.487, 1.369, 0.000)],
        ["N9", 0, (2.533, 0.000, 0.000)],
        ["C1'", 0, (0.000, 0.000, 0.000)],
        ["O5'", 0, (2.010, 1.328, 0.000)],
        ["P", 0, (2.237, 1.370, 0.000)],
        ["O3'", 0, (1.997, 2.078, 0.000)],
        ["C4", 0, (1.316, 0.639, 0.000)],
    ],
    'C': [
        ["C5'", 0, (1.509, 0.000, 0.000)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (-0.487, 1.367, 0.000)],
        ["N1", 0, (2.467, 0.000, 0.000)],
        ["C1'", 0, (0.000, 0.000, 0.000)],
        ["O5'", 0, (2.009, 1.329, 0.000)],
        ["P", 0, (2.171, 1.598, 0.000)],
        ["O3'", 0, (1.968, 1.560, 0.000)],
        ["C2", 0, (1.281, 0.733, 0.000)],
    ],
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "A": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "OP3", "N1", "N3", "N6", "N7", "C2", "C5", "C6", "C8"],
    "U": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "OP3", "N3", "C4", "C5", "C6", "O2", "O4"],
    "G": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "OP3", "N1", "N2", "N3", "N7", "C2", "C5", "C6", "C8", "O6"],
    "C": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "OP3", "N3", "N4", "C4", "C5", "C6", "O2"],
}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).

atom_types = [
    "C1'", 
    "C2'", 
    "C3'", 
    "C4'", 
    "C5'", 
    "O5'", 
    "O4'", 
    "O3'", 
    "O2'", 
    "P", 
    "OP1",
    "OP2",
    "OP3",
    "N1", 
    "N2", 
    "N3", 
    "N4", 
    "N6", 
    "N7", 
    "N9", 
    "C2", 
    "C4", 
    "C5", 
    "C6", 
    "C8", 
    "O2", 
    "O4", 
    "O6",
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28.

# Format: The list for each NT type contains gamma, beta, alpha, chi in
# this order. 
chi_angles_atoms = {
    'A': [
        ["O4'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "O3'"],
        ["O4'", "C1'", "N9", "C4"],
    ],
    'U': [
        ["O4'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "O3'"],
        ["O4'", "C1'", "N1", "C2"],
    ],
    'G': [
        ["O4'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "O3'"],
        ["O4'", "C1'", "N9", "C4"],
    ],
    'C': [
        ["O4'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "O3'"],
        ["O4'", "C1'", "N1", "C2"],
    ]
}

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom9_names = {
    "A": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4"],
    "U": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2"],
    "G": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4"],
    "C": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2"],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see above).
# In the order of gamma, beta, alpha, chi
chi_angles_mask = [
    [1.0, 1.0, 1.0, 1.0],  # A
    [1.0, 1.0, 1.0, 1.0],  # U
    [1.0, 1.0, 1.0, 1.0],  # G
    [1.0, 1.0, 1.0, 1.0],  # C
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
# Noted that none of the chi angles are pi periodic in RNA
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # A
    [0.0, 0.0, 0.0, 0.0],  # U
    [0.0, 0.0, 0.0, 0.0],  # G
    [0.0, 0.0, 0.0, 0.0],  # C
    [0.0, 0.0, 0.0, 0.0],  # Unknown
]

# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Define chi-angle rigid groups via one-hot representations."""
    chi_angles_index = {}
    one_hots = []

    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (4 - len(indices)))
        chi_angles_index[k] = indices

    for r in restypes:
        # Adopt one-letter format in RNA
        # res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[r]]
        one_hots.append(one_hot)

    one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])

    return one_hot


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group

# 5 NT type, 28 atom type
restype_atom28_to_rigid_group = np.zeros([5, 28], dtype=np.int)
# 5 NT type, 28 atom type
restype_atom28_mask = np.zeros([5, 28], dtype=np.float32)
# 5 NT type, 28 atom type, 3 position
restype_atom28_rigid_group_positions = np.zeros([5, 28, 3], dtype=np.float32)
# 5 NT type, 9 atom type
restype_atom9_to_rigid_group = np.zeros([5, 9], dtype=np.int)
# 5 NT type, 9 atom type
restype_atom9_mask = np.zeros([5, 9], dtype=np.float32)
# 5 NT type, 9 atom type, 3 position
restype_atom9_rigid_group_positions = np.zeros([5, 9, 3], dtype=np.float32)
# 5 NT type, 4 torsion, 4*4 tensor 
# restype_rigid_group_default_frame = np.zeros([5, 4, 4, 4], dtype=np.float32)

def _make_rigid_group_constants():
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(restypes):
        # resname = restype_1to3[restype_letter]
        ntname = restype_letter
        for atomname, group_idx, atom_position in rigid_group_atom_positions[
            ntname
        ]:
            atomtype = atom_order[atomname]
            restype_atom28_to_rigid_group[restype, atomtype] = group_idx
            restype_atom28_mask[restype, atomtype] = 1
            restype_atom28_rigid_group_positions[
                restype, atomtype, :
            ] = atom_position

            atom9idx = restype_name_to_atom9_names[ntname].index(atomname)
            restype_atom9_to_rigid_group[restype, atom9idx] = group_idx
            restype_atom9_mask[restype, atom9idx] = 1
            restype_atom9_rigid_group_positions[
                restype, atom9idx, :
            ] = atom_position


_make_rigid_group_constants()
