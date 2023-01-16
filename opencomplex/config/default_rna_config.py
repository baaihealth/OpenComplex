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

import ml_collections as mlc
from opencomplex.config.references import *

config = mlc.ConfigDict({
    "data": {
        "common": {
            "feat": {
                "hmm": [NUM_RES, None],
                "ss": [NUM_RES, NUM_RES, None],
                "dis_n": [NUM_RES, NUM_RES, None],
                "dis_c4": [NUM_RES, NUM_RES, None],
                "dis_p": [NUM_RES, NUM_RES, None],
                "omg": [NUM_RES, NUM_RES, None],
                "theta": [NUM_RES, NUM_RES, None],
                "eta_bb": [NUM_RES, None],
                "theta_bb": [NUM_RES, None],
                "chi": [NUM_RES, None],
                **common_feats,
            },
            "masked_msa": {
                "profile_prob": 0.1,
                "same_prob": 0.1,
                "uniform_prob": 0.1,
            },
            "max_recycling_iters": 3,
            "msa_cluster_features": False,
            "reduce_msa_clusters_by_max_templates": False,
            "resample_msa_in_recycling": True,
            "unsupervised_features": [
                "butype",
                "residue_index",
                "seq_length",
                "between_segment_residues",
                "deletion_matrix",
                "no_recycling_iters",
            ],
            "use_templates": False,
            "complex_type": "RNA",
            "c_butype": 4,
        },
        "supervised": {
            "clamp_prob": 0.9,
            "supervised_features": [
                "all_atom_mask",
                "all_atom_positions",
                "resolution",
                "ss",
                "msa",
                "hmm",
                "seq",
                "use_clamped_fape",
                "is_distillation",
                # geometry_head
                'dis_n', 
                'dis_c4', 
                'dis_p', 
                'omg', 
                'theta',
                # torsion_head
                'eta_bb', 
                'theta_bb', 
                'chi',
            ],
        },
        "predict": {
            "fixed_size": False,
            "subsample_templates": False,  # We want top templates.
            "masked_msa_replace_fraction": 0.0,
            "max_msa_clusters": 512,
            "max_template_hits": 0,
            "max_templates": 0,
            "max_extra_msa": 5120,
            "crop": False,
            "crop_size": None,
            "supervised": False,
            "uniform_recycling": False,
        },
        "eval": {
            "fixed_size": False,
            "subsample_templates": False,  # We want top templates.
            "masked_msa_replace_fraction": 0.0,
            "max_msa_clusters": 128,
            "max_template_hits": 0,
            "max_templates": 0,
            "max_extra_msa": 5120,
            "crop": False,
            "crop_size": None,
            "supervised": True,
            "uniform_recycling": False,
        },
        "train": {
            "fixed_size": False,
            "subsample_templates": False,
            "masked_msa_replace_fraction": 0.0,
            "max_extra_msa": 5120,
            "max_msa_clusters": 128,
            "max_template_hits": 0,
            "max_templates": 0,
            "shuffle_top_k_prefiltered": 20,
            "crop": True,
            "crop_size": None,
            "supervised": True,
            "clamp_prob": 0.9,
            "max_distillation_msa_clusters": 1000,
            "uniform_recycling": True,
            "distillation_prob": 0.75,
        },
        "data_module": {
            "data_loaders": {
                "batch_size": 1,
                "num_workers": 8,
            },
        }
    },
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
        "c_butype": 4,
        "c_m": c_m,
        "c_z": c_z,
        "c_s": c_s,
        "no_single_transformer_blocks": 4,
        "no_pair_transformer_blocks": 2,
        "_mask_trans": False,
        "extra_msa": {
            "enabled": False,
        },
        "evoformer_stack": {
            "c_m": c_m,
            "c_z": c_z,
            "c_s": c_s,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "c_hidden_msa_att": 16,
            "c_hidden_opm": 16,
            "c_hidden_mul": 32,
            "c_hidden_pair_att": 16,
            "no_heads_msa": 8,
            "no_heads_pair": 8,
            "no_blocks": 16,
            "transition_n": 16,
            # set at 1 when training, else None
            "blocks_per_ckpt": 1,
            "clear_cache_between_blocks": False,
            "inf": 1e9,
            "eps": 1e-10,  # 1e-10,
        },
        "pair_transformer_stack": {
            "c_z": c_z,
            "pair_dropout": 0.25,
            "c_hidden_mul": 32,
            "c_hidden_pair_att": 16,
            "no_heads_pair": 8,
            "no_blocks": 2,
            "transition_n": 2,
            # set at 1 when training, else None
            "blocks_per_ckpt": 1,
            "clear_cache_between_blocks": False,
            "inf": 1e9,
            "eps": 1e-10,  # 1e-10,
        },
        "input_embedder": {
            "tf_dim": 6,
            "msa_dim": 9,
            "c_z": c_z,
            "c_m": c_m,
            "max_relative_idx": 32,
        },
        "hmm_embedder": {
            "c_in": 15,
            "c_m": c_m,
        },
        "ss_embedder": {
            "c_in": 1,
            "c_z": c_z,
        },
        "recycling_embedder": {
            "c_m": c_m,
            "c_z": c_z,
            "min_bin": 1.5,
            "max_bin": 41.5,
            "no_bins": 20,
            "inf": 1e8,
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
            "epsilon": 1e-12,  # 1e-12,
            "inf": 1e5,
        },
        "heads": {
            "torsion_head": {
                "c_s": c_s,
                "hidden_dim": 64,
                "no_blocks": 2,
                "no_angles": 3,
                "angle_bins": 24,
            },
            "geometry_head": {
                "c_z": c_z,
                # "min_dist": 2.0,
                # "max_dist": 40.0,
                "dist_bins": 38+2,
                "omega_bins": 24+1,
                "theta_bins": 24+1,
                "phi_bins": 12+1,
            },
            "masked_msa": {
                "c_m": c_m,
                "c_out": 7,
            },
            "lddt": {
                # TODO: check the number of bins
                "no_bins": 25,
                "c_in": c_s,
                "c_hidden": 128,
            },
            "distogram": {
                "c_z": c_z,
                "no_bins": aux_distogram_bins,
            },
            "experimentally_resolved": {
                "c_s": c_s,
                "c_out": 27,
            },
            "tm": {
                "c_z": c_z,
                "no_bins": aux_distogram_bins,
                "enabled": tm_enabled,
            },
        },
        "template": {
            "enabled": False,
            "embed_angles": False,
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
    },
    "relax": {
        "max_iterations": 0,  # no max
        "tolerance": 2.39,
        "stiffness": 10.0,
        "max_outer_iterations": 20,
        "exclude_residues": [],
    },
    "loss": {
        "disgeometry": {
            "weight": 0.5,
        },
        "distorsion": {
            "weight": 0.5,
        },
        "disbackbone": {
            "weight": 0.6,
            "eps": 1e-8,  # 1e-8,
        },
        "angbackbone": {
            "weight": 0.6,
            "eps": 1e-8,  # 1e-8,
            "tolerance_factor_soft": 0.0,
            "tolerance_factor_hard": 12.0,
        },
        "fape": {
            "backbone": {
                "clamp_distance": 15.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
                "enable_use_clamped_fape": False,
            },
            "sidechain": {
                "clamp_distance": 15.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "eps": 1e-4,
            "weight": 3.0,
        },
         "plddt_loss": {
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "cutoff": 15.0,
            "no_bins": 25,
            "eps": 1e-8,  # 1e-10,
            "weight": 0.01,
        },
        "masked_msa": {
            "eps": 1e-8,  # 1e-8,
            # original weight is set at 2
            "weight": 2.0,
        },
        "supervised_chi": {
            "chi_weight": 0.5,
            "angle_norm_weight": 0.01,
            "eps": 1e-8,  # 1e-6,
            "weight": 1.0,
        },
        "violation": {
            "violation_tolerance_factor": 0.0,
            "clash_overlap_tolerance": 1.5,
            "eps": 1e-8,  # 1e-6,
            "weight": 0.0,
        },
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
        "tm": {
            "max_bin": 31,
            "no_bins": 64,
            "min_resolution": 0.1,
            "max_resolution": 3.0,
            "eps": eps,  # 1e-8,
            "weight": 0.,
            "enabled": tm_enabled,
        },
        "eps": 1e-8,
    },
    "ema": {"decay": 0.999},
})
