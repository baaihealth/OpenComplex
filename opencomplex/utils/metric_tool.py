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
from opencomplex.np import protein
from opencomplex.data import mmcif_parsing
from opencomplex.np import residue_constants, nucleotide_constants
from opencomplex.utils.tensor_utils import (
    tensor_tree_map
)
from opencomplex.utils.loss import lddt_ca, lddt, lddt_all
from opencomplex.utils.loss_rna import lddt_O4, compute_drmsd
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
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

class MetricTool:
    @classmethod
    def compute_metric(cls, native, prediction, fast=False):
        UNKNOWN = 20

        metrics = {}

        gt_coords = native["all_atom_positions"]
        gt_mask = native["all_atom_mask"]

        # remove unknown residues
        gt_butype = torch.argmax(native["butype"], dim=-1)
        gt_coords = gt_coords[torch.where(gt_butype != UNKNOWN)]
        gt_mask = gt_mask[torch.where(gt_butype != UNKNOWN)]

        pred_coords = prediction["all_atom_positions"]
        all_atom_mask = gt_mask * prediction["all_atom_mask"]

        ca_pos = residue_constants.atom_order["CA"]

        gt_coords_ca = gt_coords[..., ca_pos, :]
        pred_coords_ca = pred_coords[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

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
                pred_coords,
                gt_coords,
                all_atom_mask,
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

    @classmethod
    def compute_dockq(
        cls,
        native_file_path: str,
        prediciton_file_path: str,
    ):
        cif_file = native_file_path + ".cif"
        native_file_path += ".pdb"
        if not os.path.exists(native_file_path):
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("", cif_file)
            io = PDBIO()
            io.set_structure(structure)
            io.save(native_file_path)



    @classmethod
    def compute_all_metric(
        cls,
        native_dir: str,
        prediction_dir: str,
        mode: str = "monomer",
        target_list: list = None,
    ):
        prediction_list = MetricTool.get_prediction_list(prediction_dir, target_list)

        keys_to_use = ["all_atom_positions", "all_atom_mask", "butype"]

        current_chain = current_gt = None
        current_sequences = None
        logging.info("compute metrics for %d inference results...", len(prediction_list))
        for target, prediction_file_path in tqdm.tqdm(prediction_list):
            metric_file = os.path.join(os.path.dirname(prediction_file_path), target + "_metric.json")
            if os.path.exists(metric_file):
                logging.info('metric %s exists, skip', metric_file)
                continue
            try:
                splitted = target.split("_")
                if mode == "monomer":
                    protein_name = splitted[0]
                    chain_id = splitted[1]
                    chain = f"{protein_name}_{chain_id}"
                    setting = "_".join(splitted[2:])

                    if chain != current_chain:
                        current_chain = chain
                        with open(os.path.join(native_dir, protein_name + ".cif"), "r", encoding="utf-8") as f:
                            mmcif_string = f.read()
                        mmcif_object = mmcif_parsing.parse(file_id=protein_name, mmcif_string=mmcif_string).mmcif_object

                        current_gt = make_mmcif_features(mmcif_object, chain_id=chain_id)
                        current_gt = {
                            k: torch.tensor(v)
                            for k, v in current_gt.items()
                            if k in keys_to_use
                        }

                    with open(prediction_file_path, "r", encoding="utf-8") as f:
                        pdb_string = f.read()
                    protein_object = protein.from_pdb_string(pdb_string)
                    prediction = make_protein_features(protein_object, target)
                    prediction = {
                        k: torch.tensor(v)
                        for k, v in prediction.items()
                        if k in keys_to_use
                    }

                    metric = MetricTool.compute_metric(current_gt, prediction)
                elif mode == "multimer":
                    protein_name = chain = splitted[0]
                    setting = "_".join(splitted[1:])
                    with open(prediction_file_path, "r", encoding="utf-8") as f:
                        pdb_string = f.read()
                    protein_object = protein.from_pdb_string(pdb_string)
                    prediction = make_protein_features(protein_object, target)
                    prediction = {
                        k: torch.tensor(v)
                        for k, v in prediction.items()
                        if k in keys_to_use
                    }

                    chain_index = np.sort(protein_object.chain_index)
                    unique_chain_index = np.unique(chain_index).tolist()
                    sequences = []
                    sequences_with_prediction = []
                    seq_counter = defaultdict(int)
                    for i, idx in enumerate(unique_chain_index):
                        seq = residue_constants.aatype_to_str_sequence(protein_object.aatype[chain_index==idx])
                        sequences.append(seq)
                        seq_counter[seq] += 1
                        
                        sequences_with_prediction.append((
                            i,
                            seq,
                            prediction["all_atom_positions"][chain_index==idx],
                            prediction["all_atom_mask"][chain_index==idx]))

                    if chain != current_chain or ''.join(sequences) != current_sequences:
                        current_chain = chain
                        current_sequences = ''.join(sequences)
                        with open(os.path.join(native_dir, protein_name + ".cif"), "r", encoding="utf-8") as f:
                            mmcif_string = f.read()
                        mmcif_object = mmcif_parsing.parse(file_id=protein_name, mmcif_string=mmcif_string).mmcif_object
                        chains = copy.deepcopy(mmcif_object.chain_to_seqres)
                        chain_id = []
                        for seq in sequences:
                            for c in list(chains.keys()):
                                if chains[c] == seq:
                                    chain_id.append(c)
                                    del chains[c]
                                    break
                        current_gt = {}
                        for c in chain_id:
                            feat = make_mmcif_features(mmcif_object, c)
                            for k, v in feat.items():
                                if k not in current_gt:
                                    current_gt[k] = v
                                elif v.ndim > 0:
                                    current_gt[k] = np.concatenate((current_gt[k], v), axis=0)
                        current_gt = {
                            k: torch.tensor(v)
                            for k, v in current_gt.items()
                            if k in keys_to_use
                        }
                    # find best permutation metric
                    best_metric = best_prediction = None
                    total_permutations = 1
                    for _, v in seq_counter.items():
                        total_permutations *= math.factorial(v)
                    for _ in tqdm.tqdm(range(total_permutations)):
                        permutation_prediction = {
                            "all_atom_positions": torch.cat([x[2] for x in sequences_with_prediction]),
                            "all_atom_mask": torch.cat([x[3] for x in sequences_with_prediction]),
                        }
                        metric = MetricTool.compute_metric(current_gt, permutation_prediction, fast=True)
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
                    metric = MetricTool.compute_metric(current_gt, best_prediction)

                metric["chain"] = chain
                metric["setting"] = setting
                metric_file = os.path.join(os.path.dirname(prediction_file_path), target + "_metric.json")
                with open(metric_file, "w", encoding="utf-8") as f:
                    json.dump(metric, f, indent=4)
            except Exception as e:
                print(target)
                print(e)
                import traceback as tb
                tb.print_exc()


    @classmethod
    def save_metrics(cls, all_metrics, output_dir):
        with open(os.path.join(output_dir, "metrics_verbose.json"), "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4)
        summary_metrics = defaultdict(lambda: defaultdict(list))
        for _, chain_metric in all_metrics.items():
            for setting, metrics in chain_metric.items():
                for k, v in metrics.items():
                    summary_metrics[setting][k].append(v)

        for setting, metrics in summary_metrics.items():
            for k, v in metrics.items():
                metrics[k] = statistics.fmean(v)

        with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=4)

    @classmethod
    def summarize_metrics(cls, base_dir):
        summary_metrics = defaultdict(lambda: defaultdict(list))
        for root, _, f_names in os.walk(base_dir):
            for f_name in f_names:
                if f_name.endswith('metric.json'):
                    with open(os.path.join(root, f_name), "r", encoding="utf-8") as f:
                        metric = json.load(f)
                    setting = metric.pop("setting")
                    metric.pop("chain")

                    for k, v in metric.items():
                        summary_metrics[setting][k].append(v)

        for setting, metrics in summary_metrics.items():
            for k, v in metrics.items():
                metrics[k] = statistics.fmean(v)

        with open(os.path.join(base_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=4)


    @classmethod
    def get_prediction_list(cls, prediction_dir, target_list):
        prediction_list = []
        for root, _, f_names in os.walk(prediction_dir):
            prediction_list.extend([
                (f.split(".", maxsplit=1)[0], os.path.join(root, f))
                for f in f_names if f.endswith(".pdb")])

        prediction_list = sorted(prediction_list)
        if target_list is not None:
            def in_list(p, target_list):
                return any(map(lambda t: t in p[0], target_list))
            prediction_list = [p for p in prediction_list if in_list(p, target_list)]

        return prediction_list

class ValidationMetrics:
    @classmethod
    def protein_metrics(cls,
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]

        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]

        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-8,
            per_residue=False,
        )

        metrics["lddt_ca"] = lddt_ca_score

        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )

        metrics["drmsd_ca"] = drmsd_ca_score

        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

        return metrics

    @classmethod
    def RNA_metrics(cls,
        batch,
        outputs,
        superimposition_metrics=False
    ):
        metrics = {}

        # B, N, 23, 3
        gt_coords_23 = batch["atom23_gt_positions"]
        # B, N, 23, 3
        pred_coords_23 = outputs["sm"]['positions'][-1][0].unsqueeze(dim=0)
        # pred_coords = outputs["final_atom_positions"]
        # B, N, 27
        all_atom_mask = batch["all_atom_mask"]
        # B, N, 23
        atom23_mask = batch["atom23_gt_exists"]
        
        gt_coords_masked = gt_coords_23 * atom23_mask[..., None]
        pred_coords_masked = pred_coords_23 * atom23_mask[..., None]
        
        measured_atom_order = ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N", "O2'", "Cb"]
        
        for i in range(len(measured_atom_order)):
            ca_pos = i
            gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
            pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
            atom_mask_ca = atom23_mask[..., ca_pos]

            drmsd_ca_score = compute_drmsd(
                pred_coords_masked_ca,
                gt_coords_masked_ca,
                mask=atom_mask_ca,
            )

            metrics["drmsd_{}".format(measured_atom_order[i])] = drmsd_ca_score.to(torch.float32)
            
        lddt_O4_score = lddt_O4(
            # 23 atom
            pred_coords_masked,
            # 23 atom
            gt_coords_masked,
            # no used here
            all_atom_mask.new_ones(pred_coords_masked.shape[:-1]),
            eps=1e-8,
            per_residue=False,
        )

        metrics["lddt_O4"] = lddt_O4_score.to(torch.float32)
        # pos of O4', index in 23-atom format
        ca_pos = 2
        gt_coords_ca = gt_coords_23[..., ca_pos, :]
        pred_coords_ca = pred_coords_23[..., ca_pos, :]
        atom_mask_ca = atom23_mask[..., ca_pos]

        if(superimposition_metrics):
            superimposed_pred, _ = superimpose(
                gt_coords_ca, pred_coords_ca, atom_mask_ca
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, atom_mask_ca
            )

            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

        return metrics