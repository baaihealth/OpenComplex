"""Constants used in RNAFold."""

import collections
import functools
from typing import Mapping, List, Tuple
from importlib import resources

import numpy as np
import tree

# Distance from one C4 to next C4 [trans configuration: omega = 180].
c4_c4 = 6.12

# Format: The list for each NT type contains delta, gamma, beta, alpha1, alpha1, tm, chi in this order. 
chi_angles_atoms = {
    'A': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    'U': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ],
    'G': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    'C': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ]
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order.
# In the order of delta, gamma, beta, alpha1, alpha1, tm, chi
chi_angles_mask = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # A
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # U
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # G
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # C
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
# Noted that none of the chi angles are pi periodic in RNA
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # A
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # U
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # G
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # C
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Unknown
]


# Atoms positions relative to the rigid groups, defined by the beta, gamma, delta, and chi.
# 0: 'backbone group 1',
# 1: 'backbone group 2',
# 2: 'delta-group',
# 3: 'gamma-group',
# 4: 'beta-group',
# 5: 'alpha1-group',
# 6: 'alpha2-group',
# 7: 'tm-group',
# 8: 'chi-group',
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]

rigid_group_atom_positions = {
    'A': [
        ["C3'", 0, (-0.378, 1.475, 0.00)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.508, -0.803, -1.174)],
        ["C2'", 1, (-0.448, 1.458, 0.00)],
        ["C1'", 1, (0.000, 0.000, 0.000)],
        ["N9",  1, (-0.477, -0.746, 1.17)],
        ["O3'", 2, (0.524, 1.321, 0.00)],
        ["O5'", 3, (0.511, 1.333, -0.0)],
        ["P",   4, (0.817, 1.367, -0.0)],
        ["OP1", 5, (0.470, 1.407, 0.00)],
        ["OP2", 6, (0.464, 1.409, -0.0)],
        ["O2'", 7, (0.467, 1.335, -0.0)],
        ["N1",  8, (2.807, 2.869, 0.002)],
        ["N3",  8, (0.446, 2.395, -0.008)],
        ["N6",  8, (4.438, 1.239, 0.022)],
        ["N7",  8, (2.110, -0.769, 0.013)],
        ["C2",  8, (1.510, 3.194, -0.006)],
        ["C4",  8, (0.817, 1.104, 0.000)],
        ["C5",  8, (2.108, 0.616, 0.009)],
        ["C6",  8, (3.146, 1.563, 0.011)],
        ["C8",  8, (0.838, -1.084, 0.008)],
    ],
    'U': [
        ["C3'", 0, (-0.378, 1.474, 0.00)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.514, -0.806, -1.17)],
        ["C2'", 1, (-0.445, 1.461, 0.00)],
        ["C1'", 1, (0.000, 0.000, 0.000)],
        ["N1",  1, (-0.493, -0.756, 1.171)],
        ["O3'", 2, (0.522, 1.322, 0.00)],
        ["O5'", 3, (0.514, 1.332, -0.0)],
        ["P",   4, (0.820, 1.364, -0.0)],
        ["OP1", 5, (0.462, 1.408, -0.0)],
        ["OP2", 6, (0.466, 1.408, 0.00)],
        ["O2'", 7, (0.473, 1.333, -0.0)],
        ["N3",  8, (2.018, 1.154, -0.0)],
        ["C2",  8, (0.649, 1.221, 0.00)],
        ["C4",  8, (2.79, 0.014, -0.001)],
        ["C5",  8, (2.05, -1.21, -0.004)],
        ["C6",  8, (0.714, -1.175, -0.003)],
        ["O2",  8, (0.06, 2.288, -0.001)],
        ["O4",  8, (4.015, 0.113, 0.002)],
    ],
    'G': [
        ["C3'", 0, (-0.369, 1.476, 0.00)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.513, -0.806, -1.171)],
        ["C2'", 1, (-0.448, 1.458, 0.00)],
        ["C1'", 1, (0.000, 0.000, 0.000)],
        ["N9",  1, (-0.488, -0.737, 1.171)],
        ["O3'", 2, (0.529, 1.319, 0.00)],
        ["O5'", 3, (0.514, 1.331, -0.0)],
        ["P",   4, (0.814, 1.367, -0.0)],
        ["OP1", 5, (0.472, 1.406, 0.00)],
        ["OP2", 6, (0.464, 1.408, -0.0)],
        ["O2'", 7, (0.472, 1.334, -0.0)],
        ["N1",  8, (2.750, 2.841, -0.006)],
        ["N2",  8, (1.216, 4.548, 0.001)],
        ["N3",  8, (0.415, 2.391, 0.005)],
        ["N7",  8, (2.096, -0.776, -0.013)],
        ["C2",  8, (1.437, 3.232, -0.0)],
        ["C4",  8, (0.818, 1.104, 0.00)],
        ["C5",  8, (2.102, 0.61, -0.007)],
        ["C6",  8, (3.186, 1.523, -0.009)],
        ["C8",  8, (0.830, -1.092, -0.01)],
        ["O6",  8, (4.394, 1.274, -0.014)],
    ],
    'C': [
        ["C3'", 0, (-0.372, 1.476, 0.00)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.517, -0.809, -1.17)],
        ["C2'", 1, (-0.449, 1.459, 0.00)],
        ["C1'", 1, (0.000, 0.000, 0.000)],
        ["N1",  1, (-0.502, -0.76, 1.164)],
        ["O3'", 2, (0.528, 1.322, 0.00)],
        ["O5'", 3, (0.517, 1.332, -0.0)],
        ["P",   4, (0.818, 1.364, 0.0)],
        ["OP1", 5, (0.469, 1.407, -0.0)],
        ["OP2", 6, (0.469, 1.408, 0.00)],
        ["O2'", 7, (0.476, 1.333, -0.0)],
        ["N3",  8, (2.036, 1.22, 0.001)],
        ["N4",  8, (4.036, 0.115, 0.003)],
        ["C2",  8, (0.683, 1.220, 0.0)],
        ["C4",  8, (2.706, 0.067, -0.002)],
        ["C5",  8, (2.036, -1.188, -0.008)],
        ["C6",  8, (0.698, -1.175, -0.009)],
        ["O2",  8, (0.039, 2.276, 0.001)],
    ],
}


# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "A": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "N1", "N3", "N6", "N7", "C2", "C5", "C6", "C8"],
    "U": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "G": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "N1", "N2", "N3", "N7", "C2", "C5", "C6", "C8", "O6"],
    "C": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "N3", "N4", "C4", "C5", "C6", "O2"],
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "P": 1.8,
}

Bond = collections.namedtuple(
    "Bond", ["atom1_name", "atom2_name", "length", "stddev"]
)
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)

@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[
    Mapping[str, List[Bond]],
    Mapping[str, List[Bond]],
    Mapping[str, List[BondAngle]],
]:
    """Load stereo_chemical_props.txt into a nice structure.

    Load literature values for bond lengths and bond angles and translate
    bond angles into the length of the opposite edge of the triangle
    ("residue_virtual_bonds").

    Returns:
      residue_bonds:  dict that maps resname --> list of Bond tuples
      residue_virtual_bonds: dict that maps resname --> list of Bond tuples
      residue_bond_angles: dict that maps resname --> list of BondAngle tuples
    """
    stereo_chemical_props = resources.read_text("opencomplex.resources", "stereo_chemical_props_RNA.txt")

    lines_iter = iter(stereo_chemical_props.splitlines())
    # Load bond lengths.
    residue_bonds = {}
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(
            Bond(atom1, atom2, float(length), float(stddev))
        )
    residue_bonds["X"] = []

    # Load bond angles.
    residue_bond_angles = {}
    next(lines_iter)  # Skip empty line.
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split("-")
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["X"] = []

    def make_bond_key(atom1_name, atom2_name):
        """Unique key to lookup bonds."""
        return "-".join(sorted([atom1_name, atom2_name]))

    # Translate bond angles into distances ("virtual bonds").
    residue_virtual_bonds = {}
    for resname, bond_angles in residue_bond_angles.items():
        # Create a fast lookup dict for bond lengths.
        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            # Compute distance between atom1 and atom3 using the law of cosines
            # c^2 = a^2 + b^2 - 2ab*cos(gamma).
            gamma = ba.angle_rad
            length = np.sqrt(
                bond1.length ** 2
                + bond2.length ** 2
                - 2 * bond1.length * bond2.length * np.cos(gamma)
            )

            # Propagation of uncertainty assuming uncorrelated errors.
            dl_outer = 0.5 / length
            dl_dgamma = (
                2 * bond1.length * bond2.length * np.sin(gamma)
            ) * dl_outer
            dl_db1 = (
                2 * bond1.length - 2 * bond2.length * np.cos(gamma)
            ) * dl_outer
            dl_db2 = (
                2 * bond2.length - 2 * bond1.length * np.cos(gamma)
            ) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2
                + (dl_db1 * bond1.stddev) ** 2
                + (dl_db2 * bond2.stddev) ** 2
            )
            residue_virtual_bonds[resname].append(
                Bond(ba.atom1_name, ba.atom3name, length, stddev)
            )

    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)


between_res_bond_length_o3_p = 1.602
between_res_bond_length_stddev_o3_p = 0.001

between_res_cos_angles_c3_o3_p = [-0.5030, 0.9867]  # degrees: 120.197 +- 9.3694
between_res_cos_angles_o3_p_o5 = [-0.2352, 0.9932]  # degrees: 103.602 +- 6.7053
between_res_cos_angles_o4_c1_n = [-0.3322, 0.9979]  # degrees: 109.402 +- 3.70
between_res_cos_angles_c1_n_c =  [-0.5383, 0.9988]  # degrees: 122.570 +- 2.8078
between_res_cos_angles_c1_c2_c3 = [-0.2006, 0.9999]  # degrees: 101.575 +- 0.9811
between_res_cos_angles_c2_c3_c4 = [-0.2126, 0.9996]  # degrees: 102.277 +- 1.6858

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
atom_type_num = len(atom_types)  # := 27.


# A compact atom encoding with 23 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom23_names = {
    "A": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N9", "O2'", "C4", "N1", "N3", "N6", "N7", "C2", "C5", "C6", "C8", ""],
    "U": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N1", "O2'", "C2", "N3", "C4", "C5", "C6", "O2", "O4", "", "", ""],
    "G": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N9", "O2'", "C4", "N1", "N2", "N3", "N7", "C2", "C5", "C6", "C8", "O6"],
    "C": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N1", "O2'", "C2", "N3", "N4", "C4", "C5", "C6", "O2", "", "", ""],
}


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "G",
    "C",
    "U"
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


def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    # +1 to account for unknown (all 0s).
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        # restype_name = restype_1to3[restype_letter]
        restype_name = restype_letter
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask

STANDARD_ATOM_MASK = _make_standard_atom_mask()


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


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    mat = np.stack(
        [ex_normalized, ey_normalized, eznorm, translation]
    ).transpose()
    mat = np.concatenate([mat, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return mat


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group

# 5 NT type, 27 atom type
nttype_atom27_to_rigid_group = np.zeros([5, 27], dtype=np.int)
# 5 NT type, 27 atom type
nttype_atom27_mask = np.zeros([5, 27], dtype=np.float32)
# 5 NT type, 27 atom type, 3 position
nttype_atom27_rigid_group_positions = np.zeros([5, 27, 3], dtype=np.float32)
# 5 NT type, 23 atom type
nttype_atom23_to_rigid_group = np.zeros([5, 23], dtype=np.int)
# 5 NT type, 23 atom type
nttype_atom23_mask = np.zeros([5, 23], dtype=np.float32)
# 5 NT type, 23 atom type, 3 position
nttype_atom23_rigid_group_positions = np.zeros([5, 23, 3], dtype=np.float32)
# 5 NT type, 9 groups, 4*4 tensor 
nttype_rigid_group_default_frame = np.zeros([5, 9, 4, 4], dtype=np.float32)

def _make_rigid_group_constants():
    """Fill the arrays above."""
    for nttype, nttype_letter in enumerate(restypes):
        # resname = restype_1to3[restype_letter]
        ntname = nttype_letter
        for atomname, group_idx, atom_position in rigid_group_atom_positions[
            ntname
        ]:
            atomtype = atom_order[atomname]
            nttype_atom27_to_rigid_group[nttype, atomtype] = group_idx
            nttype_atom27_mask[nttype, atomtype] = 1
            nttype_atom27_rigid_group_positions[
                nttype, atomtype, :
            ] = atom_position

            atom9idx = restype_name_to_atom23_names[ntname].index(atomname)
            nttype_atom23_to_rigid_group[nttype, atom9idx] = group_idx
            nttype_atom23_mask[nttype, atom9idx] = 1
            nttype_atom23_rigid_group_positions[
                nttype, atom9idx, :
            ] = atom_position

    for nttype, nttype_letter in enumerate(restypes):
        # resname = restype_1to3[restype_letter]
        ntname = nttype_letter
        atom_positions = {
            name: np.array(pos)
            for name, _, pos in rigid_group_atom_positions[ntname]
        }

        # backbone1 to backbone1 is the identity transform
        nttype_rigid_group_default_frame[nttype, 0, :, :] = np.eye(4)

        # backbone2 to backbone2 is the identity transform
        nttype_rigid_group_default_frame[nttype, 1, :, :] = np.eye(4)

        # delta-frame to backbone1
        if chi_angles_mask[nttype][0]:
            base_atom_names = chi_angles_atoms[ntname][0]
            base_atom_positions = [
                atom_positions[name] for name in base_atom_names
            ]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 2, :, :] = mat
            
        # gamma-frame to backbone1
        if chi_angles_mask[nttype][1]:
            base_atom_names = chi_angles_atoms[ntname][1]
            base_atom_positions = [
                atom_positions[name] for name in base_atom_names
            ]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 3, :, :] = mat
        
        # beta-frame to gamma-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        if chi_angles_mask[nttype][2]:
            axis_end_atom_name = chi_angles_atoms[ntname][2][2]
            axis_end_atom_position = atom_positions[axis_end_atom_name]
            mat = _make_rigid_transformation_4x4(
                ex=axis_end_atom_position,
                ey=np.array([-1.0, 0.0, 0.0]),
                translation=axis_end_atom_position,
            )
            nttype_rigid_group_default_frame[nttype, 4, :, :] = mat

        # alpha1-frame to beta-frame
        # alpha2-frame to beta-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for torsion_idx in range(3, 5):
            if chi_angles_mask[nttype][torsion_idx]:
                axis_end_atom_name = chi_angles_atoms[ntname][torsion_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                nttype_rigid_group_default_frame[
                    nttype, 2 + torsion_idx, :, :
                ] = mat
                
        # tm-frame to backbone2
        if chi_angles_mask[nttype][5]:
            base_atom_names = chi_angles_atoms[ntname][5]
            base_atom_positions = [
                atom_positions[name] for name in base_atom_names
            ]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 7, :, :] = mat
        
        # chi-frame to backbone2
        if chi_angles_mask[nttype][6]:
            base_atom_names = chi_angles_atoms[ntname][6]
            base_atom_positions = [
                atom_positions[name] for name in base_atom_names
            ]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 8, :, :] = mat

_make_rigid_group_constants()


def make_atom23_dists_bounds(
    overlap_tolerance=1.5, bond_length_tolerance_factor=15
):
    """compute upper and lower bounds for bonds to assess violations."""
    restype_atom23_bond_lower_bound = np.zeros([5, 23, 23], np.float32)
    restype_atom23_bond_upper_bound = np.zeros([5, 23, 23], np.float32)
    restype_atom23_bond_stddev = np.zeros([5, 23, 23], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):
        # resname = restype_1to3[restype_letter]
        resname = restype_letter
        atom_list = restype_name_to_atom23_names[resname]

        # create lower and upper bounds for clashes
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10
                restype_atom23_bond_lower_bound[
                    restype, atom1_idx, atom2_idx
                ] = lower
                restype_atom23_bond_lower_bound[
                    restype, atom2_idx, atom1_idx
                ] = lower
                restype_atom23_bond_upper_bound[
                    restype, atom1_idx, atom2_idx
                ] = upper
                restype_atom23_bond_upper_bound[
                    restype, atom2_idx, atom1_idx
                ] = upper

        # overwrite lower and upper bounds for bonds and angles
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom23_bond_lower_bound[
                restype, atom1_idx, atom2_idx
            ] = lower
            restype_atom23_bond_lower_bound[
                restype, atom2_idx, atom1_idx
            ] = lower
            restype_atom23_bond_upper_bound[
                restype, atom1_idx, atom2_idx
            ] = upper
            restype_atom23_bond_upper_bound[
                restype, atom2_idx, atom1_idx
            ] = upper
            restype_atom23_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom23_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {
        "lower_bound": restype_atom23_bond_lower_bound,  # shape (5,23,23)
        "upper_bound": restype_atom23_bond_upper_bound,  # shape (5,23,23)
        "stddev": restype_atom23_bond_stddev,  # shape (5,23,23)
    }