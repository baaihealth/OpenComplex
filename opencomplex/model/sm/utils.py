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
