import ml_collections as mlc

c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_FULL_RES = "num residues before crop placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

common_feats = mlc.ConfigDict({
    "butype": [NUM_RES],
    "all_atom_mask": [NUM_RES, None],
    "all_atom_positions": [NUM_RES, None, None],
    "alt_chi_angles": [NUM_RES, None],
    "dense_atom_alt_gt_exists": [NUM_RES, None],
    "dense_atom_alt_gt_positions": [NUM_RES, None, None],
    "dense_atom_exists": [NUM_RES, None],
    "dense_atom_is_ambiguous": [NUM_RES, None],
    "dense_atom_gt_exists": [NUM_RES, None],
    "dense_atom_gt_positions": [NUM_RES, None, None],
    "all_atom_exists": [NUM_RES, None],
    "backbone_rigid_mask": [NUM_RES, None],
    "backbone_rigid_tensor": [NUM_RES, None, None, None],
    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
    "chi_angles_sin_cos": [NUM_RES, None, None],
    "chi_mask": [NUM_RES, None],
    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
    "is_distillation": [],
    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
    "msa_row_mask": [NUM_MSA_SEQ],
    "no_recycling_iters": [],
    "pseudo_beta": [NUM_RES, None],
    "pseudo_beta_mask": [NUM_RES],
    "residue_index": [NUM_RES],
    "residx_dense_to_all": [NUM_RES, None],
    "residx_all_to_dense": [NUM_RES, None],
    "resolution": [],
    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
    "rigidgroups_group_exists": [NUM_RES, None],
    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
    "rigidgroups_gt_exists": [NUM_RES, None],
    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
    "seq_length": [],
    "seq_mask": [NUM_RES],
    "target_feat": [NUM_RES, None],
    "template_butype": [NUM_TEMPLATES, NUM_RES],
    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
    "template_all_atom_positions": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_alt_torsion_angles_sin_cos": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
    "template_backbone_rigid_tensor": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "template_mask": [NUM_TEMPLATES],
    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
    "template_sum_probs": [NUM_TEMPLATES, None],
    "template_torsion_angles_mask": [
        NUM_TEMPLATES, NUM_RES, None,
    ],
    "template_torsion_angles_sin_cos": [
        NUM_TEMPLATES, NUM_RES, None, None,
    ],
    "true_msa": [NUM_MSA_SEQ, NUM_RES],
    "use_clamped_fape": [],
})

multimer_feats = mlc.ConfigDict({
    "sym_id": [NUM_RES],
    "asym_id": [NUM_RES],
    "entity_id": [NUM_RES],

    "origin_all_atom_mask": [NUM_FULL_RES, None],
    "origin_all_atom_positions": [NUM_FULL_RES, None, None],
    "origin_sym_id": [NUM_FULL_RES],
    "origin_asym_id": [NUM_FULL_RES],
    "origin_entity_id": [NUM_FULL_RES],

    "pseudo_beta_mask": [NUM_FULL_RES],
    "rigidgroups_gt_exists": [NUM_FULL_RES, None],
    "dense_atom_gt_exists": [NUM_FULL_RES, None],
    "rigidgroups_gt_frames": [NUM_FULL_RES, None, None, None],
    "rigidgroups_group_exists": [NUM_FULL_RES, None],
    "dense_atom_alt_gt_exists": [NUM_FULL_RES, None],
    "backbone_rigid_tensor": [NUM_FULL_RES, None, None, None],
    "pseudo_beta": [NUM_FULL_RES, None],
    "dense_atom_alt_gt_positions": [NUM_FULL_RES, None, None],
    "backbone_rigid_mask": [NUM_FULL_RES, None],
    "chi_mask": [NUM_FULL_RES, None],
    "rigidgroups_alt_gt_frames": [NUM_FULL_RES, None, None, None],
    "rigidgroups_group_is_ambiguous": [NUM_FULL_RES, None],
    "chi_angles_sin_cos": [NUM_FULL_RES, None, None],
    "dense_atom_gt_positions": [NUM_FULL_RES, None, None],
    "dense_atom_is_ambiguous": [NUM_FULL_RES, None],

    "resolution": [None],
})

__all__ = [
    "c_z", "c_m", "c_t", "c_e", "c_s", "blocks_per_ckpt", "chunk_size", "aux_distogram_bins", "tm_enabled",
    "eps", "templates_enabled", "embed_template_torsion_angles", "tune_chunk_size",
    "NUM_RES", "NUM_FULL_RES", "NUM_MSA_SEQ", "NUM_EXTRA_SEQ", "NUM_TEMPLATES",
    "common_feats", "multimer_feats"
]