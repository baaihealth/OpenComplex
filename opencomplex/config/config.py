import copy
import importlib
import ml_collections as mlc

from opencomplex.config.references import *
from opencomplex.config import default_protein_config, default_rna_config, default_complex_config

def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting[p]

        return setting

    mutually_exclusive_bools = [
        (
            "model.template.average_templates", 
            "model.template.offload_templates"
        ),
        (
            "globals.use_lma",
            "globals.use_flash",
        ),
    ]

    for s1, s2 in mutually_exclusive_bools:
        s1_setting = string_to_setting(s1)
        s2_setting = string_to_setting(s2)
        if(s1_setting and s2_setting):
            raise ValueError(f"Only one of {s1} and {s2} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if(config.globals.use_flash and not fa_is_installed):
        raise ValueError("use_flash requires that FlashAttention is installed")


def model_config(
    name, 
    train=False, 
    low_prec=False, 
    long_sequence_inference=False
):
    if "rna" in name.lower():
        c = copy.deepcopy(default_rna_config.config)
    elif "multimer" in name.lower():
        c = copy.deepcopy(default_protein_config.multimer_config)
    elif "mix" in name.lower():
        c = copy.deepcopy(default_complex_config.config)
    else:
        c = copy.deepcopy(default_protein_config.config)

    
    # TRAINING PRESETS
    if name == "initial_training":
        # AF2 Suppl. Table 4, "initial training" setting
        pass
    elif name == "finetuning":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
    elif name == "finetuning_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.train.crop_size = 384
        c.data.train.max_msa_clusters = 512
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "finetuning_no_templ":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.model.template.enabled = False
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
    elif name == "finetuning_no_templ_ptm":
        # AF2 Suppl. Table 4, "finetuning" setting
        c.data.train.crop_size = 384
        c.data.train.max_extra_msa = 5120
        c.data.train.max_msa_clusters = 512
        c.model.template.enabled = False
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    # INFERENCE PRESETS
    elif name == "model_1":
        # AF2 Suppl. Table 5, Model 1.1.1
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
    elif name == "model_2":
        # AF2 Suppl. Table 5, Model 1.1.2
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
    elif name == "model_3":
        # AF2 Suppl. Table 5, Model 1.2.1
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
    elif name == "model_4":
        # AF2 Suppl. Table 5, Model 1.2.2
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
    elif name == "model_5":
        # AF2 Suppl. Table 5, Model 1.2.3
        c.model.template.enabled = False
    elif name == "model_1_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120 
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_2_ptm":
        c.data.common.reduce_max_clusters_by_max_templates = True
        c.data.common.use_templates = True
        c.data.common.use_template_torsion_angles = True
        c.model.template.enabled = True
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_3_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_4_ptm":
        c.data.train.max_extra_msa = 5120
        c.data.predict.max_extra_msa = 5120
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_5_ptm":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_1_multimer_v2":
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
        c.model.input_embedder.tf_dim = 21

        c.data.common.chi_angle_only = True
        c.model.structure_module.split_kv = True
        c.model.structure_module.trans_scale_factor = 20
        c.model.evoformer_stack.outer_product_mean_first = True
        c.model.extra_msa.extra_msa_stack.outer_product_mean_first = True
        c.model.template.template_pair_embedder.use_separate_linear = True
        c.model.template.template_angle_embedder.use_alt_torsion_angles = False
        c.model.template.template_angle_embedder.c_in = 34
    elif name == "fast":
        c.model.evoformer_stack.no_blocks = 16
        c.data.train.crop_size = 128

        c.data.train.max_extra_msa = 1024
        c.data.eval.max_extra_msa = 1024
        c.data.predict.max_extra_msa = 1024

        c.data.train.max_extra_msa = 1024

        c.data.train.max_msa_clusters = 128
        c.data.eval.max_msa_clusters = 128
        c.data.predict.max_msa_clusters = 128

        c.data.common.max_recycling_iters = 3
    elif name == "multimer_fast":
        c.model.evoformer_stack.no_blocks = 16
        c.data.train.crop_size = 128

        c.data.train.max_extra_msa = 1024
        c.data.eval.max_extra_msa = 1024
        c.data.predict.max_extra_msa = 1024

        c.data.train.max_extra_msa = 1024

        c.data.train.max_msa_clusters = 128
        c.data.eval.max_msa_clusters = 128
        c.data.predict.max_msa_clusters = 128

        c.data.common.max_recycling_iters = 3

        c.data.data_module.data_loaders.num_workers = 16
    elif name == "rna":
        c_z.set(32)
        c_m.set(32)
        c_s.set(64)
    elif name == "mix":
        c.model.evoformer_stack.no_blocks = 16

        c.data.train.crop_size = 128

        c.data.train.max_msa_clusters = 128
        c.data.eval.max_msa_clusters = 128
        c.data.predict.max_msa_clusters = 128

        c.data.common.max_recycling_iters = 3
    elif name == "mix_finetune":
        c.model.evoformer_stack.no_blocks = 16

        c.data.train.crop_size = 128

        c.data.train.max_msa_clusters = 128
        c.data.eval.max_msa_clusters = 128
        c.data.predict.max_msa_clusters = 128

        c.data.common.max_recycling_iters = 3

        c.loss.supervised_chi.weight = 2.0
    elif name == "mix_large":
        c.model.evoformer_stack.no_blocks = 48

        c.data.train.crop_size = 384

        c.data.train.max_msa_clusters = 256 
        c.data.eval.max_msa_clusters = 256
        c.data.predict.max_msa_clusters = 256

        c.data.common.max_recycling_iters = 3
    else:
        raise ValueError("Invalid model name")

    if long_sequence_inference:
        assert(not train)
        c.globals.offload_inference = True
        c.globals.use_lma = True
        c.globals.use_flash = False
        c.model.template.offload_inference = True
        c.model.template.template_pair_stack.tune_chunk_size = False
        c.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        c.model.evoformer_stack.tune_chunk_size = False
    
    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None
        c.globals.use_lma = False
        c.globals.offload_inference = False
        c.model.template.average_templates = False
        c.model.template.offload_templates = False
    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    enforce_config_constraints(c)

    return c
