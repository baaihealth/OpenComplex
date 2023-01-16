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

from opencomplex.data import templates, feature_pipeline, data_pipeline
from opencomplex.config import model_config, NUM_RES
from opencomplex.model.model import OpenComplex
from opencomplex.model.torchscript import script_preset_
from opencomplex.np import residue_constants, protein
from opencomplex.utils.metric_tool import MetricTool
import opencomplex.np.relax.relax as relax
from opencomplex.utils.import_weights import (
    import_jax_weights_,
)
from opencomplex.utils.tensor_utils import (
    tensor_tree_map,
)
from opencomplex.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
import tqdm
from scripts.utils import add_data_args


TRACING_INTERVAL = 50

def get_target_list(args, infer_from_fasta):
    if infer_from_fasta:
        target_list = [target.split('.')[:-1] for target in os.listdir(dir) if target.endswith((".fasta", ".fa"))]
    else:
        feature_exist = lambda target: os.path.exists(os.path.join(args.features_dir, target, "features.pkl"))
        target_list = [target for target in os.listdir(args.features_dir) if feature_exist(target)]

    if args.target_list_file is not None:
        if os.path.exists(args.target_list_file):
            with open(args.target_list_file, "r", encoding="utf-8") as f:
                specified_target_list = f.read().splitlines()
            target_list = [target for target in specified_target_list if target in target_list]
        else:
            logger.warning("target list file %s provided but not exist.", args.target_list_file)

    return sorted(target_list)
        

def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        if(args.use_precomputed_alignments is None and not os.path.isdir(local_alignment_dir)):
            logger.info(f"Generating alignments for {tag}...")
                
            os.makedirs(local_alignment_dir)

            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                no_cpus=args.cpus,
            )
            alignment_runner.run(
                tmp_fasta_path, local_alignment_dir
            )
        else:
            logger.info(
                f"Using precomputed alignments for {tag} at {alignment_dir}..."
            )

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def run_model(model, batch, target):
    with torch.no_grad(): 
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any([
            "template_" in k for k in batch
        ])

        logger.info(f"Running inference for {target}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
   
        model.config.template.enabled = template_enabled

    return out


def prep_output(out, batch, feature_dict, feature_processor, config_preset, args):
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
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
        f"config_preset={config_preset}",
    ])

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / args.multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if(c != cur_chain):
            cur_chain = c
            prev_chain_max = i + cur_chain * args.multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path, super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def load_models_from_command_line(args, model_device):
    # Create the output directory

    config_presets = args.config_presets
    param_paths = args.param_paths

    mode = "monomer"

    num_models = len(config_presets)
    assert num_models == len(param_paths), "Different number of configs and parameter files provided!"
    multiple_model_mode = num_models > 1

    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    ret = []
    for config_preset, path in zip(config_presets, param_paths):
        config = model_config(config_preset)
        model = OpenComplex(config=config, complex_type=args.complex_type)
        model = model.eval()

        feature_processor = feature_pipeline.FeaturePipeline(config.data)

        # pytorch checkpoint
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
        output_directory = make_output_directory(args.output_dir, checkpoint_basename, multiple_model_mode)
        
        model = model.to(model_device)
        ret.append((config_preset, config, model, feature_processor, output_directory))

        if "multimer" in config_preset:
            mode = "multimer"

    return ret, mode


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def get_feature_dict(target, data_processor, alignment_dir, infer_from_fasta, args):
    if infer_from_fasta:
        # Gather input sequences
        filename = os.path.join(args.fasta_dir, f"{target}.fa")
        if not os.path.exists(filename):
            filename += "sta"
        with open(filename, "r", encoding="utf-8") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)
        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        # Does nothing if the alignments have already been computed
        precompute_alignments(tags, seqs, alignment_dir, args)

        feature_dict = generate_feature_dict(
            tags,
            seqs,
            alignment_dir,
            data_processor,
            args,
        )
    else:
        feature_dict = data_processor.process_prepared_features(os.path.join(args.features_dir, target))

    return feature_dict

def main(worker_id, args, infer_from_fasta):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    target_list = get_target_list(args, infer_from_fasta)[worker_id::args.num_workers]
    print('my workid is: %s, my target list is: %s' % (worker_id, target_list))
    
    template_featurizer = alignment_dir = None
    if infer_from_fasta:    
        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )

        if args.use_precomputed_alignments is None:
            alignment_dir = os.path.join(output_dir_base, "alignments")
        else:
            alignment_dir = args.use_precomputed_alignments
        
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    _get_feature_dict = partial(
        get_feature_dict,
        data_processor=data_processor,
        alignment_dir=alignment_dir,
        infer_from_fasta=infer_from_fasta,
        args=args)

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    
    if args.use_gpu:
        model_device = f"cuda:{worker_id}"
        torch.cuda.set_device(model_device)
    else:
        model_device = "cpu"
    logger.info("my model device is %s", model_device)
    models, mode = load_models_from_command_line(args, model_device)    

    for target in tqdm.tqdm(target_list):
        cur_tracing_interval = 0
        feature_dict = None
        for config_preset, config, model, feature_processor, output_directory in models:
            output_name = f'{target}_{config_preset}'
            unrelaxed_output_path = os.path.join(
                output_directory, f'{output_name}_unrelaxed.pdb'
            )
            if args.output_postfix is not None:
                output_name = f'{output_name}_{args.output_postfix}'
    
            if not args.overwrite and os.path.exists(unrelaxed_output_path):
                logger.info(f"Inference result %s already exists, skip...", unrelaxed_output_path)
                continue

            if feature_dict is None:
                feature_dict = _get_feature_dict(target)
                if(args.trace_model):
                    n = feature_dict["butype"].shape[-2]
                    rounded_seqlen = round_up_seqlen(n)
                    feature_dict = pad_feature_dict_seq(
                        feature_dict, rounded_seqlen,
                    )

            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict',
            )

            processed_feature_dict = {
                k:torch.as_tensor(v, device=model_device) 
                for k,v in processed_feature_dict.items()
            }

            if(args.trace_model):
                if(rounded_seqlen > cur_tracing_interval):
                    logger.info(
                        f"Tracing model at {rounded_seqlen} residues..."
                    )
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        f"Tracing time: {tracing_time}"
                    )
                    cur_tracing_interval = rounded_seqlen

            out = run_model(model, processed_feature_dict, target)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()), 
                processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            logger.info("plddt: %.5f", np.mean(out['plddt']))

            unrelaxed_protein = prep_output(
                out, 
                processed_feature_dict, 
                feature_dict, 
                feature_processor, 
                config_preset,
                args
            )

            with open(unrelaxed_output_path, 'w') as fp:
                fp.write(protein.to_pdb(unrelaxed_protein))

            logger.info(f"Output written to {unrelaxed_output_path}...")
            
            if not args.skip_relaxation:
                amber_relaxer = relax.AmberRelaxation(
                    use_gpu=(model_device != "cpu"),
                    **config.relax,
                )

                # Relax the prediction.
                logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                t = time.perf_counter()
                visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
                if "cuda" in model_device:
                    device_no = model_device.split(":")[-1]
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_no
                try:
                    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
                    relaxation_time = time.perf_counter() - t

                    logger.info(f"Relaxation time: {relaxation_time}")

                    # Save the relaxed PDB.
                    relaxed_output_path = os.path.join(
                        output_directory, f'{output_name}_relaxed.pdb'
                    )
                    with open(relaxed_output_path, 'w') as fp:
                        fp.write(relaxed_pdb_str)
                    
                    logger.info(f"Relaxed output written to {relaxed_output_path}...")
                except ValueError as e:
                    logger.warn(f"Cannot relax {target}")
                    print(e)

            if args.save_outputs:
                output_dict_path = os.path.join(
                    output_directory, f'{output_name}_output_dict.pkl'
                )
                with open(output_dict_path, "wb") as fp:
                    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(f"Model output written to {output_dict_path}...")

    if args.native_dir is not None:
        MetricTool.compute_all_metric(args.native_dir, args.output_dir, mode, target_list)

def update_timings(dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """Write dictionary of one or more run step times to a file"""
    import json
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_dir", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
    )
    parser.add_argument(
        "--features_dir", type=str,
        help="Directory containing processed feature files in pkl format. Named as `<target_name>/features.pkl`"
    )
    parser.add_argument(
        "--target_list_file", type=str,
        help="Path to a txt file with each line is a name of target to infer."
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
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
        "--config_presets", nargs="+", default=[],
        help="""Name of a model config preset defined in opencomplex/config.py"""
    )
    parser.add_argument(
        "--param_paths", nargs="+", default=[],
        help="Paths to opencomplex checkpoint. Should have the same order with config_presets."
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="""Number of workers to run in parallel. If use_gpu is True, num_workers should be less than
                total number of available GPUs."""
    )
    parser.add_argument(
        "--overwrite", default=False, action="store_true",
        help="Whether overwrite existing inference result."
    )
    parser.add_argument(
        "--native_dir", type=str,
        help="Directory to ground truth mmcif fils. If provided, metrics will be computed."
    )
    parser.add_argument(
        "--complex_type", type=str, default="protein", choices=["protein", "RNA"],
    )
    
    add_data_args(parser)
    args = parser.parse_args()

    if(not args.use_gpu and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if args.use_gpu and args.num_workers > torch.cuda.device_count():
        raise ValueError("Num workers should not be greater than device count if using gpu.")

    if args.fasta_dir is not None:
        logging.info("Inference from Fasta files")
        infer_from_fasta = True
    elif args.features_dir is not None:
        logging.info("Inference from processed feature files")
        infer_from_fasta = False
    else:
        logging.error("No input files provided!")

    _worker = partial(main, args=args, infer_from_fasta=infer_from_fasta)
    mp.set_start_method("spawn")
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as p:
            p.map(_worker, range(args.num_workers))
        p.join()
    else:
        _worker(0)


    if args.native_dir is not None:
        MetricTool.summarize_metrics(args.output_dir)