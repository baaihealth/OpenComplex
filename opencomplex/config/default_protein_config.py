import copy
import ml_collections as mlc
from opencomplex.config.references import *

config = mlc.ConfigDict({
    "data": {
        "common": {
            "feat": common_feats,
            "masked_msa": {
                "profile_prob": 0.1,
                "same_prob": 0.1,
                "uniform_prob": 0.1,
            },
            "max_recycling_iters": 3,
            "msa_cluster_features": True,
            "reduce_msa_clusters_by_max_templates": False,
            "resample_msa_in_recycling": True,
            "template_features": [
                "template_all_atom_positions",
                "template_sum_probs",
                "template_butype",
                "template_all_atom_mask",
            ],
            "unsupervised_features": [
                "butype",
                "residue_index",
                "msa",
                "num_alignments",
                "seq_length",
                "between_segment_residues",
                "deletion_matrix",
                "no_recycling_iters",
            ],
            "use_templates": templates_enabled,
            "use_template_torsion_angles": embed_template_torsion_angles,
            "complex_type": "protein",
            "c_butype": 20,
        },
        "supervised": {
            "clamp_prob": 0.9,
            "supervised_features": [
                "all_atom_mask",
                "all_atom_positions",
                "resolution",
                "use_clamped_fape",
                "is_distillation",
            ],
        },
        "predict": {
            "fixed_size": True,
            "subsample_templates": False,  # We want top templates.
            "masked_msa_replace_fraction": 0.15,
            "max_msa_clusters": 512,
            "max_extra_msa": 1024,
            "max_template_hits": 4,
            "max_templates": 4,
            "crop": False,
            "crop_size": None,
            "supervised": False,
            "uniform_recycling": False,
        },
        "eval": {
            "fixed_size": True,
            "subsample_templates": False,  # We want top templates.
            "masked_msa_replace_fraction": 0.15,
            "max_msa_clusters": 128,
            "max_extra_msa": 1024,
            "max_template_hits": 4,
            "max_templates": 4,
            "crop": False,
            "crop_size": None,
            "supervised": True,
            "uniform_recycling": False,
        },
        "train": {
            "fixed_size": True,
            "subsample_templates": True,
            "masked_msa_replace_fraction": 0.15,
            "max_msa_clusters": 128,
            "max_extra_msa": 1024,
            "max_template_hits": 4,
            "max_templates": 4,
            "shuffle_top_k_prefiltered": 20,
            "crop": True,
            "crop_size": 256,
            "supervised": True,
            "clamp_prob": 0.9,
            "max_distillation_msa_clusters": 1000,
            "uniform_recycling": True,
            "distillation_prob": 0.75,
        },
        "data_module": {
            "use_small_bfd": False,
            "data_loaders": {
                "batch_size": 1,
                "num_workers": 8,
            },
        },
    },
    # Recurring FieldReferences that can be changed globally here
    "globals": {
        "blocks_per_ckpt": blocks_per_ckpt,
        "chunk_size": chunk_size,
        # Use Staats & Rabe's low-memory attention algorithm. Mutually
        # exclusive with use_flash.
        "use_lma": False,
        # Use FlashAttention in selected modules. Mutually exclusive with 
        # use_lma. Doesn't work that well on long sequences (>1000 residues).
        "use_flash": False,
        "offload_inference": False,
        "c_z": c_z,
        "c_m": c_m,
        "c_t": c_t,
        "c_e": c_e,
        "c_s": c_s,
        "eps": eps,
    },
    "model": {
        "_mask_trans": False,
        "c_butype": 20,
        "input_embedder": {
            "tf_dim": 22,
            "msa_dim": 49,
            "c_z": c_z,
            "c_m": c_m,
            "max_relative_idx": 32,
        },
        "recycling_embedder": {
            "c_z": c_z,
            "c_m": c_m,
            "min_bin": 3.25,
            "max_bin": 20.75,
            "no_bins": 15,
            "inf": 1e8,
        },
        "template": {
            "distogram": {
                "min_bin": 3.25,
                "max_bin": 50.75,
                "no_bins": 39,
            },
            "template_angle_embedder": {
                # DISCREPANCY: c_in is supposed to be 51.
                "c_in": 57,
                "c_out": c_m,
            },
            "template_pair_embedder": {
                "c_in": 88,
                "c_out": c_t,
            },
            "template_pair_stack": {
                "c_t": c_t,
                # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                # as 64. In the code, it's 16.
                "c_hidden_tri_att": 16,
                "c_hidden_tri_mul": 64,
                "no_blocks": 2,
                "no_heads": 4,
                "pair_transition_n": 2,
                "dropout_rate": 0.25,
                "blocks_per_ckpt": blocks_per_ckpt,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
            },
            "template_pointwise_attention": {
                "c_t": c_t,
                "c_z": c_z,
                # DISCREPANCY: c_hidden here is given in the supplement as 64.
                # It's actually 16.
                "c_hidden": 16,
                "no_heads": 4,
                "inf": 1e5,  # 1e9,
            },
            "inf": 1e5,  # 1e9,
            "eps": eps,  # 1e-6,
            "enabled": templates_enabled,
            "embed_angles": embed_template_torsion_angles,
            "use_unit_vector": False,
            # Approximate template computation, saving memory.
            # In our experiments, results are equivalent to or better than
            # the stock implementation. Should be enabled for all new
            # training runs.
            "average_templates": False,
            # Offload template embeddings to CPU memory. Vastly reduced
            # memory consumption at the cost of a modest increase in
            # runtime. Useful for inference on very long sequences.
            # Mutually exclusive with average_templates.
            "offload_templates": False,
        },
        "extra_msa": {
            "extra_msa_embedder": {
                "c_in": 25,
                "c_out": c_e,
            },
            "extra_msa_stack": {
                "c_m": c_e,
                "c_z": c_z,
                "c_hidden_msa_att": 8,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 4,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
                "ckpt": blocks_per_ckpt is not None,
            },
            "enabled": True,
        },
        "evoformer_stack": {
            "c_m": c_m,
            "c_z": c_z,
            "c_hidden_msa_att": 32,
            "c_hidden_opm": 32,
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "c_s": c_s,
            "no_heads_msa": 8,
            "no_heads_pair": 4,
            "no_blocks": 48,
            "transition_n": 4,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "blocks_per_ckpt": blocks_per_ckpt,
            "clear_cache_between_blocks": False,
            "tune_chunk_size": tune_chunk_size,
            "inf": 1e9,
            "eps": eps,  # 1e-10,
        },
        "structure_module": {
            "c_s": c_s,
            "c_z": c_z,
            "c_ipa": 16,
            "c_resnet": 128,
            "no_heads_ipa": 12,
            "no_qk_points": 4,
            "no_v_points": 8,
            "dropout_rate": 0.1,
            "no_blocks": 8,
            "no_transition_layers": 1,
            "no_resnet_blocks": 2,
            "no_angles": 7,
            "trans_scale_factor": 10,
            "epsilon": eps,  # 1e-12,
            "inf": 1e5,
        },
        "heads": {
            "lddt": {
                "no_bins": 50,
                "c_in": c_s,
                "c_hidden": 128,
            },
            "distogram": {
                "c_z": c_z,
                "no_bins": aux_distogram_bins,
            },
            "tm": {
                "c_z": c_z,
                "no_bins": aux_distogram_bins,
                "enabled": tm_enabled,
            },
            "masked_msa": {
                "c_m": c_m,
                "c_out": 23,
            },
            "experimentally_resolved": {
                "c_s": c_s,
                "c_out": 37,
            },
        },
    },
    "relax": {
        "max_iterations": 0,  # no max
        "tolerance": 2.39,
        "stiffness": 10.0,
        "max_outer_iterations": 20,
        "exclude_residues": [],
    },
    "loss": {
        "distogram": {
            "min_bin": 2.3125,
            "max_bin": 21.6875,
            "no_bins": 64,
            "eps": eps,  # 1e-6,
            "weight": 0.3,
        },
        "experimentally_resolved": {
            "eps": eps,  # 1e-8,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "weight": 0.0,
        },
        "fape": {
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
            },
            "sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 1.0,
        },
        "plddt_loss": {
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "cutoff": 15.0,
            "no_bins": 50,
            "eps": eps,  # 1e-10,
            "weight": 0.01,
        },
        "masked_msa": {
            "eps": eps,  # 1e-8,
            "weight": 2.0,
        },
        "supervised_chi": {
            "chi_weight": 0.5,
            "angle_norm_weight": 0.01,
            "eps": eps,  # 1e-6,
            "weight": 1.0,
        },
        "violation": {
            "violation_tolerance_factor": 12.0,
            "clash_overlap_tolerance": 1.5,
            "eps": eps,  # 1e-6,
            "weight": 0.0,
        },
        "tm": {
            "max_bin": 31,
            "no_bins": 64,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "eps": eps,  # 1e-8,
            "weight": 0.,
            "enabled": tm_enabled,
        },
        "eps": eps,
    },
    "ema": {"decay": 0.999},
})

config_multimer = copy.deepcopy(config)
config_multimer.update(mlc.ConfigDict({
    "data": {
        "common": {
            "feat": {
                **common_feats,
                **multimer_feats,
            },
            "unsupervised_features":
                config.data.common.unsupervised_features + \
                [
                    "asym_id",
                    "entity_id",
                    "sym_id",
                ],
        },
        "train": {
            "max_msa_clusters": 252,
            "spatial_crop_ratio": 0.5,
        },
        "predict": {
            "max_msa_clusters": 252,
            "max_extra_msa": 1152,
        },
        "eval": {
            "max_msa_clusters": 252,
        }
    },
    # Recurring FieldReferences that can be changed globally here
    "globals": {
        "mode": "multimer",
    },
    "model": {
        "input_embedder": {
            "tf_dim": 22,
            'use_chain_relative': True,
            'max_relative_chain': 2,
            'max_relative_idx': 32,
        },
        "template": {
            "template_pair_embedder": {
                "c_dgram": 39,
                "c_butype": 22,
                "c_z": c_z,
            },
            "use_unit_vector": True,
        },
        "heads": {
            "masked_msa": {
                "c_out": 22,
            },
        },
    },
    "loss": {
        "fape": {
            "intra_chain_backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
            },
            "interface_backbone": {
                "clamp_distance": 1000.0,
                "loss_unit_distance": 20.0,
            },
        },
    },
}))