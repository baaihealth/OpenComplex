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
import argparse
from copy import deepcopy
from datetime import date
import logging
import math
import numpy as np
from functools import partial
import multiprocessing as mp
import os

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)
import random
import sys
import time
import torch
import re

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or 
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    pass
    # torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from opencomplex.data import feature_pipeline, data_pipeline
from opencomplex.config.config import model_config
from opencomplex.model.model import OpenComplex
from opencomplex.np import residue_constants, protein, nucleotide_constants, rna, complex
from opencomplex.utils.complex_utils import ComplexType
import opencomplex.np.relax.relax as relax
from opencomplex.utils.tensor_utils import (
    tensor_tree_map,
)
import tqdm


def get_target_list(args):
    target_list = [target for target in os.listdir(args.features_dir)]

    if args.target_list_file is not None:
        if os.path.exists(args.target_list_file):
            with open(args.target_list_file, "r", encoding="utf-8") as f:
                specified_target_list = f.read().splitlines()
            target_list = [target for target in specified_target_list if target in target_list]
        else:
            logger.warning("target list file %s provided but not exist.", args.target_list_file)

    return sorted(target_list)

def init_worker(args, device_q):
    global model, data_processor, feature_processor, model_device, config

    if device_q is None:
        model_device = "cpu"
    else:
        model_device = f"cuda:{device_q.get()}"
        torch.cuda.set_device(model_device)

    config = model_config(args.config_preset)
    model = OpenComplex(config=config, complex_type=ComplexType[args.complex_type])
    model = model.eval()
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=None
    )
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    path = args.param_path

    checkpoint_basename = get_model_basename(path)
    if os.path.isdir(path):
        # A DeepSpeed checkpoint
        ckpt_path = os.path.join(
            args.output_dir,
            checkpoint_basename + ".pt",
        )

        if not os.path.isfile(ckpt_path):
            convert_zero_checkpoint_to_fp32_state_dict(
                path,
                ckpt_path,
            )
        d = torch.load(ckpt_path)
        model.load_state_dict(d["ema"]["params"])
    else:
        ckpt_path = path
        d = torch.load(ckpt_path)

        if "ema" in d:
            # The public weights have had this done to them already
            d = d["ema"]["params"]
        model.load_state_dict(d)
    logger.info(
        f"Loaded opencomplex parameters at {path}..."
    )
    model = model.to(model_device)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)


def prep_output(out, batch, feature_dict, feature_processor, args):
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    
    if args.complex_type != 'RNA':
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
    else:
        plddt_b_factors = np.repeat(
            plddt[..., None], nucleotide_constants.atom_type_num, axis=-1
        )
    
    if(args.subtract_plddt):
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if(feature_processor.config.common.use_templates and "template_domain_names" in feature_dict):
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
            :feature_processor.config.predict.max_templates
        ]

        if("template_chain_index" in feature_dict):
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                :feature_processor.config.predict.max_templates
            ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={args.config_preset}",
    ])

    if args.complex_type == 'protein':
        unrelaxed_structure = protein.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors,
            remark=remark,
            parents=template_domain_names,
            parents_chain_index=template_chain_index,
        )
    elif args.complex_type == "RNA":
        unrelaxed_structure = rna.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors
        )
    else:
        unrelaxed_structure = complex.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors
        )
        
        
    return unrelaxed_structure


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def main(target, args):
    global model, data_processor, feature_processor, model_device
    output_name = target
    unrelaxed_output_path = os.path.join(
        args.output_dir, f'{output_name}_unrelaxed.pdb'
    )
    if not args.overwrite and os.path.exists(unrelaxed_output_path):
        return

    feature_dict = data_processor.process_prepared_features(os.path.join(args.features_dir, target))     

    batch = feature_processor.process_features(
        feature_dict, mode='predict',
    )

    batch = {
        k:torch.as_tensor(v, device=model_device) 
        for k,v in batch.items()
    }

    with torch.no_grad(): 
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any([
            "template_" in k for k in batch
        ])

        out = model(batch)
   
        model.config.template.enabled = template_enabled

    batch = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        batch
    )

    for key in ['final_affine_tensor', 'sm', 'geometry_head', 'torsion_head']:
        if key in out.keys():
            del(out[key])
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    unrelaxed_structure = prep_output(
        out, 
        batch, 
        feature_dict, 
        feature_processor, 
        args
    )

    if args.complex_type == 'protein':
        pdb_generator = protein.to_pdb
    elif args.complex_type == 'RNA':
        pdb_generator = rna.to_pdb
    else:
        pdb_generator = complex.to_pdb
    
    with open(unrelaxed_output_path, 'w') as fp:
        fp.write(pdb_generator(unrelaxed_structure))

    if not args.skip_relaxation:
        amber_relaxer = relax.AmberRelaxation(
            use_gpu=(model_device != "cpu"),
            **config.relax,
        )

        # Relax the prediction.
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
        if "cuda" in model_device:
            device_no = model_device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_no
        try:
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_structure)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(
                args.output_dir, f'{output_name}_relaxed.pdb'
            )
            with open(relaxed_output_path, 'w') as fp:
                fp.write(relaxed_pdb_str)
            
        except ValueError as e:
            logger.warn(f"Cannot relax {target}")
            print(e)

    if args.save_outputs:
        output_dict_path = os.path.join(
            args.output_dir, f'{output_name}_output_dict.pkl'
        )
        with open(output_dict_path, "wb") as fp:
            pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir", type=str,
        help="Directory containing processed feature files in pkl format. Named as `<target_name>/features.pkl`"
    )
    parser.add_argument(
        "--target_list_file", type=str,
        help="Path to a txt file with each line is a name of target to infer."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", default=False,
        help="""Whether run model on GPU"""
    )
    parser.add_argument(
        "--config_preset", type=str,
        help="""Name of a model config preset defined in opencomplex/config.py"""
    )
    parser.add_argument(
        "--param_path", type=str,
        help="Path to opencomplex checkpoint."
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="""Number of workers to run in parallel. If use_gpu is True, num_workers should be less than or equal to
                total number of available GPUs."""
    )
    parser.add_argument(
        "--overwrite", default=False, action="store_true",
        help="Whether overwrite existing inference result."
    )
    parser.add_argument(
        "--complex_type", type=str, default="protein", choices=["protein", "RNA", "mix"],
    )
    
    args = parser.parse_args()

    if(not args.use_gpu and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if args.use_gpu and args.num_workers > torch.cuda.device_count():
        raise ValueError("Num workers should not be greater than device count if using gpu.")

    mp.set_start_method("spawn")

    target_list = get_target_list(args)
    os.makedirs(args.output_dir, exist_ok=True)

    device_q = None
    if args.use_gpu:
        device_q = mp.Queue()
        for i in range(args.num_workers):
            device_q.put(i)

    worker = partial(main, args=args)
    with mp.Pool(args.num_workers, initializer=init_worker, initargs=(args, device_q)) as p:
        list(tqdm.tqdm(p.imap(worker, target_list), total=len(target_list)))
    p.join()
