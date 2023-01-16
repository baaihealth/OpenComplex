from collections import defaultdict
import random

from Bio.SVDSuperimposer import SVDSuperimposer
from opencomplex.config import NUM_FULL_RES
from opencomplex.np import residue_constants as rc

import torch


def get_transform(pred, gt, mask, device):
    """Superimpose gt to pred, return the rotation and translation.

    Args:
        pred (tensor): [N_res, 3]
            prediction Ca positions
        gt (tensor): [N_res, 3]
            ground truth Ca positions
        mask (tensor): [N_res]
    Returns:
        rot, trans: [3, 3], [3]
            rotation (right multiplication) and translation from gt to pred
    """
    sel = torch.where(mask == 1.)
    g_np = gt[sel].detach().cpu().numpy()
    p_np = pred[sel].detach().cpu().numpy()

    sup = SVDSuperimposer()
    sup.set(p_np, g_np)
    sup.run()
    rot, tran = sup.get_rotran()

    # Note the rot is right multiplication. So gt @ rot + trans == pred
    return torch.tensor(rot, device=device), torch.tensor(tran, device=device), sup.get_rms()


def get_chains(
    all_atom_positions,
    all_atom_mask,
    asym_id,
    entity_id,
    residue_index=None,
    all_gt_feats=None,
    feat_schema=None,
    batch_idx=0,
):
    chains = []
    unique_asym_ids = torch.unique(asym_id).tolist()
    ca_pos = rc.atom_order["CA"]

    for chain_id in unique_asym_ids:
        if chain_id == 0.:
            continue

        idx = torch.where(asym_id == chain_id)

        chain = {
            "chain_index": int(chain_id),
            "all_atom_positions": all_atom_positions[idx],
            "all_atom_mask": all_atom_mask[idx],
            "ca_atom_positions": all_atom_positions[idx][:, ca_pos],
            "ca_atom_mask": all_atom_mask[idx][:, ca_pos],
            "entity_id": int(entity_id[idx][0]),
        }
        chain["length"] = torch.sum(chain["ca_atom_mask"])
        if residue_index is not None:
            chain["residue_index"] = residue_index[idx]
        if all_gt_feats is not None:
            for k, v in all_gt_feats.items():
                if k not in feat_schema or NUM_FULL_RES not in feat_schema[k]:
                    continue
                schema = feat_schema[k]
                slices = []
                for dim_schema, dim in zip(schema, v.shape[1:]):
                    if dim_schema == NUM_FULL_RES:
                        slices.append(idx[0])
                    else:
                        slices.append(slice(0, dim))
                chain[k] = v[batch_idx][slices]
        
        chains.append(chain)

    return chains

    
def get_anchor_chain(gt_chains, pred_chains):
    candidates = []
    for pred_chain in pred_chains:
        num_homo_chains = 0
        for gt_chain in gt_chains:
            if gt_chain["entity_id"] == pred_chain["entity_id"]:
                num_homo_chains += 1
        candidates.append((
            num_homo_chains,
            -pred_chain["length"],
            pred_chain["entity_id"]
        ))

    candidates = sorted(candidates)
    choice = None
    for c in candidates:
        if -c[1] >= 8:
            choice = c[2]
            break
    if choice is None:
        choice = candidates[0][2]

    gt_anchor_chains = [c for c in gt_chains if c["entity_id"] == choice and c["length"] >= 8]
    pred_anchor_chains = [c for c in pred_chains if c["entity_id"] == choice and c["length"] >= 8]

    return random.choice(gt_anchor_chains), pred_anchor_chains
    

def find_optimal_alignment(
    pred_chains,
    gt_chains,
    rot, tran,
    pred_anchor_chain,
    gt_anchor_chain):
    """Find optimal permutation with given anchor chain.
    Permutation of pred.
    Evans, Richard, et al. BioRxiv (2021). Algoritm 4

    Args:
        chains (list of tuple): chain start and end idx
        pred_ca_positions (tensor): [N_res, 3]
        gt_ca_positions (tensor): [N_res, 3]
        mask (tensor): [N_res]
        rot (tensor): [3, 3]
        tran (tensor): 3
        gt_anchor_chain (int)
        pred_anchor_chain (int)
        entity_id (tensor): [N_res]

    Returns:
        permutation, rmsd
    """
    align = []
    for gt_chain in gt_chains:
        gt_chain["choosed"] = False

    gt_anchor_chain["choosed"] = True

    rmsd = 0
    for pred_chain in pred_chains:
        if pred_chain is pred_anchor_chain:
            align.append(gt_anchor_chain)
            continue
        min_dist = torch.inf
        choice = None
        for gt_chain in gt_chains:
            if gt_chain.get("choosed", False) or gt_chain["entity_id"] != pred_chain["entity_id"]:
                continue
            residue_index = pred_chain["residue_index"]
            mask = pred_chain["ca_atom_mask"] * gt_chain["ca_atom_mask"][residue_index]
            mask_idx = torch.where(mask == 1.)
            gt_ca_atom_positions = gt_chain["ca_atom_positions"][residue_index]
            pred_ca_atom_positions = pred_chain["ca_atom_positions"]

            dist = torch.sum(
                (
                    pred_ca_atom_positions[mask_idx].mean(dim=-2) -
                    (gt_ca_atom_positions[mask_idx] @ rot + tran).mean(dim=-2)
                ) ** 2
            )

            if choice is None or dist < min_dist:
                min_dist = dist
                choice = gt_chain

        choice["choosed"] = True
        align.append(choice)
        rmsd += min_dist
    
    rmsd = rmsd ** 0.5
    return align, rmsd


def multichain_permutation_alignment(batch, out, feat_schema):
    """Before computing multimer loss, chains with identical sequences should be permuted
    so that they are best-effort aligned.
    Evans, Richard, et al. BioRxiv (2021). Algoritm 3 & 4

    Args:
        batch (dict): batch feature with ground truth
        out (dict): prediction

    Returns:
        tensor: [batch_size, N_res]
            residue index after doing best permutation 
    """
    batch_size = batch['butype'].shape[0]
    device = batch["butype"].device

    batch_update = defaultdict(list)
    for batch_idx in range(batch_size):
        gt_chains = get_chains(
            batch['origin_all_atom_positions'][batch_idx],
            batch['origin_all_atom_mask'][batch_idx],
            batch['origin_asym_id'][batch_idx],
            batch['origin_entity_id'][batch_idx],
            all_gt_feats=batch,
            feat_schema=feat_schema,
            batch_idx=batch_idx)

        pred_chains = get_chains(
            out['final_atom_positions'][batch_idx],
            out['final_atom_mask'][batch_idx],
            batch['asym_id'][batch_idx],
            batch['entity_id'][batch_idx],
            residue_index=batch["residue_index"][batch_idx])
        gt_anchor_chain, pred_anchor_chain_candidates = get_anchor_chain(gt_chains, pred_chains)

        min_rmsd = torch.inf
        best_align = []
        for pred_chain in pred_chains:
            for gt_chain in gt_chains:
                if gt_chain["chain_index"] == pred_chain["chain_index"]:
                    best_align.append(gt_chain)
                    break
        for pred_anchor_chain in pred_anchor_chain_candidates:
            pred_chain_residue_index = pred_anchor_chain["residue_index"]
            try:
                rot, tran, rmsd_anchor = get_transform(
                    pred_anchor_chain["ca_atom_positions"],
                    gt_anchor_chain["ca_atom_positions"][pred_chain_residue_index],
                    pred_anchor_chain["ca_atom_mask"] * gt_anchor_chain["ca_atom_mask"][pred_chain_residue_index],
                    device=device
                )

                opt_align, rmsd = find_optimal_alignment(
                    pred_chains,
                    gt_chains,
                    rot, tran,
                    pred_anchor_chain,
                    gt_anchor_chain
                )
            except ZeroDivisionError:
                continue

            rmsd += rmsd_anchor
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_align = opt_align

        aligned_gt_feats = defaultdict(list)
        # print('align: ', [a["chain_index"] for a in best_align], "rmsd: ", min_rmsd)
        for i, choosed_gt_chain in enumerate(best_align):
            residue_index = pred_chains[i]["residue_index"]

            remove_keys = ["entity_id", "chain_index", "length", "choosed"]
            for k in remove_keys:
                choosed_gt_chain.pop(k, None)
            for k, v in choosed_gt_chain.items():
                aligned_gt_feats[k].append(v[residue_index])
        
        for k, v in aligned_gt_feats.items():
            if k in batch:
                batch_update[k].append(torch.cat(v))

    for k, v in batch_update.items():
        batch_update[k] = torch.stack(v)

    batch.update(batch_update)

    return batch, min_rmsd
