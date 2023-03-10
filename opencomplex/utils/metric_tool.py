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

import copy
import os
import json
import math
import logging
import statistics
from collections import defaultdict

from opencomplex.data.data_pipeline import (
    make_mmcif_features,
    make_protein_features,
)
from opencomplex.loss.loss_utils import lddt_all, lddt_ca
from opencomplex.np import protein, rna, complex
from opencomplex.data import mmcif_parsing
from opencomplex.np import residue_constants, nucleotide_constants
from opencomplex.utils.complex_utils import complex_gather, split_protein_rna_pos
from opencomplex.utils.tensor_utils import (
    padcat,
    tensor_tree_map
)
from opencomplex.utils.complex_utils import ComplexType
from opencomplex.utils.superimposition import superimpose
from opencomplex.utils.validation_metrics import (
    gdt_ts,
    gdt_ha,
    tm_score,
    drmsd
)

import torch
import numpy as np

from einops import rearrange
import tqdm

def get_rna_pos(pdb_object, complex_type):
    if complex_type == ComplexType.PROTEIN:
        return None
    elif complex_type == ComplexType.RNA:
        return list(range(pdb_object.atom_mask.shape[0]))
    elif complex_type == ComplexType.MIX:
        return list(np.where(pdb_object.chain_type == 1)[0])
    else:
        raise ValueError("Wrong complex type") 

def compute_metric_core(native, prediction, fast=False, rna_pos=None, unknown_type=20):
    metrics = {}

    gt_coords = native["all_atom_positions"]
    gt_mask = native["all_atom_mask"]

    # remove unknown residues
    gt_butype = native["butype"]
    gt_coords = gt_coords[torch.where(gt_butype != unknown_type)]
    gt_mask = gt_mask[torch.where(gt_butype != unknown_type)]

    pred_coords = prediction["all_atom_positions"]
    all_atom_mask = gt_mask * prediction["all_atom_mask"]

    ca_pos = residue_constants.atom_order["CA"]

    gt_coords_ca = gt_coords[..., ca_pos, :]
    pred_coords_ca = pred_coords[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    if rna_pos is not None:
        o4_pos = nucleotide_constants.atom_order["O4'"]
        gt_coords_ca[rna_pos, :] = gt_coords[rna_pos][:, o4_pos, :]
        pred_coords_ca[rna_pos, :] = pred_coords[rna_pos][:, o4_pos, :]
        all_atom_mask_ca[rna_pos] = all_atom_mask[rna_pos][:, o4_pos]


    metrics["drmsd_ca"] = drmsd(
        pred_coords_ca,
        gt_coords_ca,
        all_atom_mask_ca,
    )

    if not fast:
        metrics["lddt"] = lddt_all(
            pred_coords,
            gt_coords,
            all_atom_mask,
            cutoff=15,
            eps=1e-8,
            per_residue=False
        )

        metrics["lddt_ca"] = lddt_ca(
            pred_coords_ca,
            gt_coords_ca,
            all_atom_mask_ca,
            eps=1e-8,
            per_residue=False
        )

        superimposed_pred, rmsds_ca = superimpose(
            gt_coords_ca, pred_coords_ca, all_atom_mask_ca
        )


        metrics["rmsd_ca"] = rmsds_ca.float()

        _, rmsds_all = superimpose(
            rearrange(gt_coords, 'n r p -> (n r) p', p = 3),
            rearrange(pred_coords, 'n r p -> (n r) p', p = 3),
            rearrange(all_atom_mask, 'n r -> (n r)')
        )
        metrics["rmsd_all"] = rmsds_all.float()

        metrics["gdt_ts"] = gdt_ts(
            superimposed_pred, gt_coords_ca, all_atom_mask_ca
        )
        metrics["gdt_ha"] = gdt_ha(
            superimposed_pred, gt_coords_ca, all_atom_mask_ca
        )
        metrics["tm_score"] = tm_score(
            superimposed_pred, gt_coords_ca, all_atom_mask_ca
        )

    metrics = tensor_tree_map(float, metrics)
    return metrics

def compute_metric(
    native_dir: str,
    target_file: tuple,
    mode: str = "monomer",
    complex_type=ComplexType.PROTEIN,
):
    keys_to_use = ["all_atom_positions", "all_atom_mask", "butype"]

    if isinstance(complex_type, str):
        complex_type = ComplexType[complex_type]

    if complex_type == ComplexType.PROTEIN:
        pdb_parse = protein.from_pdb_string
        unknown_type = 20
    elif complex_type == ComplexType.RNA:
        pdb_parse = rna.from_pdb_string
        unknown_type = 4
    elif complex_type == ComplexType.MIX:
        pdb_parse = complex.from_pdb_string
        unknown_type = 24
    else:
        raise ValueError("Wrong complex type")

    target, prediction_file_path = target_file

    metric_file = os.path.join(os.path.dirname(prediction_file_path), target + "_metric.json")
    if os.path.exists(metric_file):
        return

    splitted = target.split("_")
    target_name = splitted[0]

    with open(prediction_file_path, "r", encoding="utf-8") as f:
        pdb_string = f.read()
    pdb_object = pdb_parse(pdb_string)
    prediction = {
        "all_atom_positions": pdb_object.atom_positions,
        "all_atom_mask": pdb_object.atom_mask
    }
    prediction = {
        k: torch.tensor(v)
        for k, v in prediction.items()
    }

    with open(os.path.join(native_dir, target_name + ".cif"), "r", encoding="utf-8") as f:
        mmcif_string = f.read()
    mmcif_object = mmcif_parsing.parse(file_id=target_name, mmcif_string=mmcif_string).mmcif_object

    if mode == "monomer":
        chain_id = splitted[1]
        chain = f"{target_name}_{chain_id}"
        setting = "_".join(splitted[2:])


        gt = make_mmcif_features(mmcif_object, chain_id=chain_id)
        gt['butype'] = np.argmax(gt['butype'], axis=-1)
        gt = {
            k: torch.tensor(v)
            for k, v in gt.items()
            if k in keys_to_use
        }
        rna_pos = get_rna_pos(pdb_object, complex_type)

        metric = compute_metric_core(gt, prediction, rna_pos=rna_pos, unknown_type=unknown_type)
    elif mode == "multimer":
        chain = target_name
        setting = "_".join(splitted[1:])
        rna_pos = get_rna_pos(pdb_object, complex_type)

        chain_index = pdb_object.chain_index
        unique_chain_index = np.unique(chain_index).tolist()
        sequences = []
        sequences_with_prediction = []
        chain_types = []
        seq_counter = defaultdict(int)
        for i, idx in enumerate(unique_chain_index):
            seq = pdb_object.get_chain_sequence(idx)
            chain_type = pdb_object.get_chain_type(idx)
            sequences.append(seq)
            chain_types.append(chain_type)
            seq_counter[seq] += 1
            
            sequences_with_prediction.append((
                i,
                seq,
                prediction["all_atom_positions"][chain_index==idx],
                prediction["all_atom_mask"][chain_index==idx]))

        chains = copy.deepcopy(mmcif_object.chain_to_seqres)
        chain_id = []
        for seq, chain_type in zip(sequences, chain_types):
            for c in list(chains.keys()):
                if chains[c] == seq or chains[c].replace('X', '') == seq:
                    chain_id.append((c, chain_type))
                    del chains[c]
                    break

        gt = {}
        for c, chain_type in chain_id:
            feat = make_mmcif_features(mmcif_object, c, chain_type=chain_type)
            feat['butype'] = np.argmax(feat['butype'], axis=-1)
            if complex_type == ComplexType.MIX:
                feat['butype'][np.where(feat['butype'] == 20)] = 24
                if chain_type == ComplexType.RNA:
                    feat['butype'] += 20
            for k, v in feat.items():
                if k not in gt:
                    gt[k] = v
                elif v.ndim > 0:
                    gt[k] = padcat((gt[k], v), axis=0)
        gt = {
            k: torch.tensor(v)
            for k, v in gt.items()
            if k in keys_to_use
        }
        # find best permutation metric
        best_metric = best_prediction = None
        total_permutations = 1
        for _, v in seq_counter.items():
            total_permutations *= math.factorial(v)
        for _ in range(total_permutations):
            permutation_prediction = {
                "all_atom_positions": torch.cat([x[2] for x in sequences_with_prediction]),
                "all_atom_mask": torch.cat([x[3] for x in sequences_with_prediction]),
            }
            metric = compute_metric_core(gt, permutation_prediction,
                                        fast=True, rna_pos=rna_pos, unknown_type=unknown_type)
            if best_metric is None or metric["drmsd_ca"] < best_metric["drmsd_ca"]:
                best_metric = metric
                best_prediction = copy.deepcopy(permutation_prediction)
            
            flag = False
            for i in range(len(sequences_with_prediction)):
                for j in range(i + 1, len(sequences_with_prediction)):
                    if sequences_with_prediction[i][0] < sequences_with_prediction[j][0] and \
                        sequences_with_prediction[i][1] == sequences_with_prediction[j][1]:
                        flag = True
                        sequences_with_prediction[i], sequences_with_prediction[j] = \
                            sequences_with_prediction[j], sequences_with_prediction[i]
                        sequences_with_prediction[:j] = sorted(sequences_with_prediction[:j])
                        break 
                if flag:
                    break
        metric = compute_metric_core(gt, best_prediction, rna_pos=rna_pos, unknown_type=unknown_type)

    metric["chain"] = chain
    metric["setting"] = setting
    metric_file = os.path.join(os.path.dirname(prediction_file_path), target + "_metric.json")
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(metric, f, indent=4)


def summarize_metrics(base_dir, target_list_file=None):
    summary_metrics = defaultdict(lambda: defaultdict(list))
    target_list = None
    if target_list_file is not None:
        target_list = open(target_list_file).read().splitlines()
    def in_list(p, target_list):
        return target_list is None or any(map(lambda t: t in p, target_list))
    for root, _, f_names in os.walk(base_dir):
        for f_name in f_names:
            if f_name.endswith('metric.json'):
                if in_list(f_name, target_list):
                    with open(os.path.join(root, f_name), "r", encoding="utf-8") as f:
                        metric = json.load(f)
                    setting = metric.pop("setting")
                    metric.pop("chain")

                    for k, v in metric.items():
                        summary_metrics[setting][k].append(v)

    for setting, metrics in summary_metrics.items():
        for k, v in metrics.items():
            metrics[k] = statistics.fmean(v)
        metrics['sample_num'] = len(v)

    with open(os.path.join(base_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)


def get_prediction_list(prediction_dir, target_list_file):
    prediction_list = []
    for root, _, f_names in os.walk(prediction_dir):
        prediction_list.extend([
            (f.split(".", maxsplit=1)[0], os.path.join(root, f))
            for f in f_names if f.endswith(".pdb")])

    prediction_list = sorted(prediction_list)
    if target_list_file is not None:
        target_list = open(target_list_file).read().splitlines()
        def in_list(p, target_list):
            return any(map(lambda t: t in p[0], target_list))
        prediction_list = [p for p in prediction_list if in_list(p, target_list)]

    return prediction_list

def compute_validation_metrics(batch, outputs, complex_type, superimposition_metrics=False):
    metrics = {}
    gt_coords = batch["all_atom_positions"]
    pred_coords = outputs["final_atom_positions"]
    all_atom_mask = batch["all_atom_mask"]

    ca_pos = residue_constants.atom_order["CA"]
    o4_pos = nucleotide_constants.atom_order["O4'"]
    protein_pos, rna_pos = split_protein_rna_pos(batch, complex_type)

    gt_coords = complex_gather(protein_pos, rna_pos, gt_coords[..., ca_pos, :], gt_coords[..., o4_pos, :], dim=-2)
    pred_coords = complex_gather(protein_pos, rna_pos, pred_coords[..., ca_pos, :], pred_coords[..., o4_pos, :], dim=-2)
    all_atom_mask = complex_gather(protein_pos, rna_pos, all_atom_mask[..., ca_pos], all_atom_mask[..., o4_pos], dim=-1)

    gt_coords = gt_coords * all_atom_mask[..., None]
    pred_coords = pred_coords * all_atom_mask[..., None]

    lddt_ca_score = lddt_all(
        pred_coords[..., None, :],
        gt_coords[..., None, :],
        all_atom_mask[..., None],
        eps=1e-8,
        per_residue=False,
    )

    metrics["lddt_ca"] = lddt_ca_score

    drmsd_ca_score = drmsd(
        pred_coords,
        gt_coords,
        mask=all_atom_mask, # still required here to compute n
    )

    metrics["drmsd_ca"] = drmsd_ca_score

    if(superimposition_metrics):
        superimposed_pred, alignment_rmsd = superimpose(
            gt_coords, pred_coords, all_atom_mask,
        )
        gdt_ts_score = gdt_ts(
            superimposed_pred, gt_coords, all_atom_mask
        )
        gdt_ha_score = gdt_ha(
            superimposed_pred, gt_coords, all_atom_mask
        )

        metrics["alignment_rmsd"] = alignment_rmsd
        metrics["gdt_ts"] = gdt_ts_score
        metrics["gdt_ha"] = gdt_ha_score

    return metrics
