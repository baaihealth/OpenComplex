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

from opencomplex.model.sm import structure_module_protein, structure_module_rna, structure_module_xyz
from opencomplex.utils.complex_utils import ComplexType


def create_structure_module(complex_type, *args, **kwargs):
    if complex_type == ComplexType.PROTEIN:
        sm = structure_module_protein.StructureModuleProtein
    elif complex_type == ComplexType.RNA:
        sm = structure_module_rna.StructureModuleRNA
    elif complex_type == ComplexType.MIX:
        sm = structure_module_xyz.StructureModuleXYZ
    else:
        raise ValueError("wrong complex type")

    return sm(*args, **kwargs)
