# Copyright 2022 BAAI
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

import enum
import torch

class CaseInsensitiveEnumMeta(enum.EnumMeta):
    def __getitem__(self, item):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)

class ComplexType(enum.Enum, metaclass=CaseInsensitiveEnumMeta):
    PROTEIN = 1
    RNA = 2
    MIX = 3

def determine_chain_type(complex_type, bio_id=None):
    if complex_type != ComplexType.MIX:
        return complex_type
    
    assert bio_id is not None

    if bio_id == 0:
        return ComplexType.PROTEIN
    else:
        return ComplexType.RNA

def correct_rna_butype(butype):
    if butype.numel() > 0 and torch.max(butype) > 7:
        return butype - 20
    return butype

def split_protein_rna_pos(bio_complex, complex_type=None):
    device = bio_complex["butype"].device

    protein_pos = []
    rna_pos = []
    if complex_type is None:
        complex_type = ComplexType.MIX if 'bio_id' in bio_complex else ComplexType.PROTEIN

    if complex_type == ComplexType.PROTEIN:
        protein_pos = torch.arange(0, bio_complex['butype'].shape[-1], device=device)
    elif complex_type == ComplexType.RNA:
        rna_pos = torch.arange(0, bio_complex['butype'].shape[-1], device=device)
    else:
        protein_pos = torch.where(bio_complex['bio_id'] == 0)[-1]
        rna_pos = torch.where(bio_complex['bio_id'] == 1)[-1]

    return protein_pos, rna_pos


def complex_gather(protein_pos, rna_pos, protein_data, rna_data, dim):
    if protein_pos is None or len(protein_pos) == 0:
        return rna_data
    elif rna_pos is None or len(rna_pos) == 0:
        return protein_data

    n = protein_data.ndim
    if dim < 0:
        dim += n
    shape = protein_data.shape
    i = protein_data.new_zeros(shape[dim:], dtype=torch.bool)
    i[protein_pos, ...] = True
    for _ in range(dim):
        i = i.unsqueeze(0)
    i = i.tile(shape[:dim] + (1,)*(n-dim))

    return torch.where(i, protein_data, rna_data)