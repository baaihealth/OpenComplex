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

import dataclasses
import io
from typing import Any, Sequence, Mapping, Optional
import re
import string

from opencomplex.np import residue_constants
from opencomplex.np import nucleotide_constants
from Bio.PDB import PDBParser
import numpy as np

from opencomplex.utils.complex_utils import ComplexType

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.
PICO_TO_ANGSTROM = 0.01

@dataclasses.dataclass(frozen=True)
class BioComplex:
    """Complex structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    butype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    chain_type: Optional[np.ndarray] = None

    def get_chain_sequence(self, idx):
        chain_type = self.chain_type[self.chain_index == idx][0]
        if chain_type == 0:
            return residue_constants.restype_to_str_sequence(self.butype[self.chain_index == idx])
        else:
            return nucleotide_constants.restype_to_str_sequence(self.butype[self.chain_index==idx])

    def get_chain_type(self, idx):
        chain_type = self.chain_type[self.chain_index == idx][0]
        return ComplexType.PROTEIN if chain_type == 0 else ComplexType.RNA


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')

          
def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> BioComplex:
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    atom_positions = []
    butype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    chain_types = []

    UNKNOWN = residue_constants.restype_num + nucleotide_constants.restype_num

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.')
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            if len(res.resname) == 3:
                # protein chain
                chain_type = 0
                res_name = residue_constants.restype_3to1.get(res.resname, 'X')
                restype_idx = residue_constants.restype_order.get(
                    res_name, UNKNOWN)
                constants = residue_constants
            else:
                chain_type = 1
                res_name = res.resname
                restype_idx = nucleotide_constants.restype_order.get(
                    res_name, UNKNOWN)
                constants = nucleotide_constants

            for atom in res:
                if atom.name not in constants.atom_types:
                    continue
                pos[constants.atom_order[atom.name]] = atom.coord
                mask[constants.atom_order[atom.name]] = 1.
                res_b_factors[constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            butype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
            chain_types.append(chain_type)


    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return BioComplex(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        butype=np.array(butype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        chain_type=np.array(chain_types),
        chain_index=chain_index,
    )


def to_pdb(bio_complex: BioComplex) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + nucleotide_constants.restypes + ['X']

    def res_1to3(r):
        if r < residue_constants.restype_num:
            return residue_constants.restype_1to3[restypes[r]]
        else:
            return restypes[r]


    pdb_lines = []

    atom_mask = bio_complex.atom_mask
    butype = bio_complex.butype
    atom_positions = bio_complex.atom_positions
    residue_index = bio_complex.residue_index.astype(np.int32)
    chain_index = bio_complex.chain_index.astype(np.int32)
    chain_type = bio_complex.chain_type.astype(np.int32)
    b_factors = bio_complex.b_factors

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
    last_chain_type = chain_type[0]
    # Add all atom sites.
    for i in range(butype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3(butype[i - 1]), chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            last_chain_type = chain_type[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(butype[i])
        atom_types = residue_constants.atom_types if last_chain_type == 0 else nucleotide_constants.atom_types
        atom_num = 37 if last_chain_type == 0 else 27


        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i][:atom_num], atom_mask[i][:atom_num], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
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
    pdb_lines.append(_chain_end(atom_index, res_1to3(butype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))

    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = False,
) -> BioComplex:
    """Assembles a complex from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """
    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if 'asym_id' in features:
        chain_index = _maybe_remove_leading_dim(features['asym_id'])
        chain_type = _maybe_remove_leading_dim(features['bio_id'])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features['butype']))
        chain_type = np.zeros_like(_maybe_remove_leading_dim(features['butype']))

    if b_factors is None:
        b_factors = np.zeros_like(result['final_atom_mask'])

    return BioComplex(
        butype=features["butype"],
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        chain_type=chain_type,
    )
