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

"""Protein data type."""
import dataclasses
import io
from typing import Any, Mapping, Optional
import re

from opencomplex.np import nucleotide_constants
import numpy as np

from opencomplex.utils.complex_utils import ComplexType

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.
PICO_TO_ANGSTROM = 0.01

# checked
@dataclasses.dataclass(frozen=True)
class RNA:
    """RNA structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # nucleotide_constants.atom_types.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Nucleotide type for each residue represented as an integer between 0 and
    # 4, where 4 is 'X'.
    nttype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    res_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the RNA that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')

    def get_chain_sequence(self, idx):
        return nucleotide_constants.restype_to_str_sequence(self.nttype[self.chain_index==idx])

    def get_chain_type(self, _):
        return ComplexType.RNA

# tmp usage
def from_input(atom_positions, atom_mask, nttype, res_index, chain_index, b_factors) -> RNA:
    
    return RNA(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        nttype=np.array(nttype),
        res_index=np.array(res_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors))


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(rna: RNA) -> str:
    """Converts a `RNA` instance to a PDB string.

    Args:
      prot: The RNA to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = nucleotide_constants.restypes + ['X']
    res_1to3 = dict(zip([i for i in range(len(restypes))], restypes))
    atom_types = nucleotide_constants.atom_types

    pdb_lines = []

    atom_mask = rna.atom_mask
    nttype = rna.nttype
    atom_positions = rna.atom_positions
    residue_index = rna.res_index.astype(np.int32)
    chain_index = rna.chain_index.astype(np.int32)
    b_factors = rna.b_factors

    if np.any(nttype > nucleotide_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    
    # Add all atom sites.
    for i in range(nttype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        # if last_chain_index != chain_index[i]:
        ## The recycling dim is not removed
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3[nttype[i - 1]], chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3[nttype[i]]
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue
            
            if np.abs(sum(pos)) < 1e-4:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # RNA supports only C, N, O, P, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                f'{residue_index[i]:>4}{insertion_code:>1}   '
                f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, res_1to3[nttype[-1]],
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def ideal_atom_mask(Rna: RNA) -> np.ndarray:
    """Computes an ideal atom mask.

    `RNA.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `RNA` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return nucleotide_constants.STANDARD_ATOM_MASK[Rna.nttype]

def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = False) -> RNA:
    """Assembles a RNA from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the RNA.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A RNA instance.
    """
    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr
    if 'asym_id' in features:
        chain_index = _maybe_remove_leading_dim(features['asym_id'])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features['butype']))

    if b_factors is None:
        b_factors = np.zeros_like(result['final_atom_mask'])

    return RNA(
        nttype=_maybe_remove_leading_dim(features['butype']),
        atom_positions=result['final_atom_positions'],
        atom_mask=result['final_atom_mask'],
        res_index=(_maybe_remove_leading_dim(features['residue_index']) + 1),
        chain_index=chain_index,
        b_factors=b_factors)
