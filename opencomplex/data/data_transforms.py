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

import itertools
from functools import partial, reduce, wraps
from operator import add
from site import ENABLE_USER_SITE

import numpy as np
import torch

from opencomplex.config.references import NUM_FULL_RES, NUM_RES, NUM_EXTRA_SEQ, NUM_TEMPLATES, NUM_MSA_SEQ
from opencomplex.np import residue_constants as rc
from opencomplex.np import nucleotide_constants as nc
from opencomplex.utils.rigid_utils import Rotation, Rigid
from opencomplex.utils.tensor_utils import (
    map_padcat,
    padcat,
    padto,
    batched_gather,
)
from opencomplex.utils.complex_utils import ComplexType, correct_rna_butype, split_protein_rna_pos


MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]

def curry1(f):
    """Supply all arguments but the first."""
    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc

def fix_data_type(bio_complex):
    keys = ["msa", "num_alignments", "num_alignments", "bio_id"]
    for k, v in bio_complex.items():
        if k in keys and v.dtype != torch.int64:
            bio_complex[k] = v.type(torch.int64)
    return bio_complex


def cast_to_64bit_ints(bio_complex):
    # We keep all ints as int64
    for k, v in bio_complex.items():
        if v.dtype == torch.int32:
            bio_complex[k] = v.type(torch.int64)

    return bio_complex


def cast_to_32bit_floats(bio_complex):
    # We keep all floats at most float32
    for k, v in bio_complex.items():
        if v.dtype == torch.float64:
            bio_complex[k] = v.type(torch.float32)

    return bio_complex


def make_one_hot(x, num_classes):
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1).type(torch.int64), 1)
    return x_one_hot


@curry1
def reorder_chains(bio_complex, shape_schema):
    """Make protein chains ahead of rna chains in complex multimer."""
    if 'bio_id' in bio_complex:
        _, indices = torch.sort(bio_complex['bio_id'], stable=True)
        for k in bio_complex:
            if k not in shape_schema:
                continue
            for i, dim_size in enumerate(shape_schema[k]):
                if dim_size == NUM_RES or dim_size == NUM_FULL_RES:
                    bio_complex[k] = torch.index_select(bio_complex[k], i, indices)
    return bio_complex

@curry1
def split_pos(bio_complex, complex_type):
    bio_complex["protein_pos"], bio_complex["rna_pos"] = split_protein_rna_pos(bio_complex, complex_type)
    return bio_complex


def map_butype(bio_complex):
    if "bio_id" in bio_complex:
        def map_token(t):
            for fr, to in [(20, 24), (21, 25)]:
                t[torch.where(t == fr)] = to
            return t

        butype_offset = 20
        rna_pos = bio_complex["rna_pos"]
        # 0~19 aa, 20~23 nt, 24 unkown, 25 gap
        bio_complex["butype"] = map_token(bio_complex["butype"])
        bio_complex['butype'][rna_pos] += butype_offset

        if "template_butype" in bio_complex:
            bio_complex["template_butype"] = map_token(bio_complex["template_butype"])
            bio_complex['template_butype'][..., rna_pos] += butype_offset

        bio_complex["msa"] = map_token(bio_complex["msa"])
        bio_complex['msa'][:, rna_pos] += butype_offset

    return bio_complex


def make_seq_mask(bio_complex):
    bio_complex["seq_mask"] = torch.ones(
        bio_complex["butype"].shape, dtype=torch.float32
    )

    # unknown type X
    bio_complex["seq_mask"][torch.where(bio_complex["butype"] == bio_complex["num_butype"])] = 0.

    return bio_complex


def make_template_mask(bio_complex):
    bio_complex["template_mask"] = torch.ones(
        bio_complex["template_butype"].shape[0], dtype=torch.float32
    )
    return bio_complex


def make_all_atom_butype(bio_complex):
    bio_complex["all_atom_butype"] = bio_complex["butype"]
    return bio_complex


@curry1
def fix_templates_butype(bio_complex, complex_type):
    # Map one-hot to indices
    if complex_type == ComplexType.RNA:
        return bio_complex

    num_templates = bio_complex["template_butype"].shape[0]
    if(num_templates > 0):
        # NOTE: data format compatibility
        if bio_complex["template_butype"].ndim != 3:
            return bio_complex

        bio_complex["template_butype"] = torch.argmax(
            bio_complex["template_butype"], dim=-1
        )
        # Map hhsearch-butype to our butype.
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        if complex_type == ComplexType.MIX:
            # NOTE(yujingcheng): magic number
            new_order_list = tuple(list(new_order_list) + list(range(22, 26)))
        new_order = torch.tensor(
            new_order_list, dtype=torch.int64, device=bio_complex["butype"].device,
        ).expand(num_templates, -1)
        bio_complex["template_butype"] = torch.gather(
            new_order, 1, index=bio_complex["template_butype"]
        )

    return bio_complex


@curry1
def correct_msa_restypes(bio_complex, complex_type):
    """Correct MSA restype to have the same order as rc."""
    if complex_type == ComplexType.RNA:
        return bio_complex

    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    if complex_type == ComplexType.MIX:
        # NOTE(yujingcheng): magic number
        new_order_list = tuple(list(new_order_list) + list(range(22, 26)))
    new_order = torch.tensor(
        [new_order_list] * bio_complex["msa"].shape[1],
        device=bio_complex["msa"].device,
    ).transpose(0, 1)
    bio_complex["msa"] = torch.gather(new_order, 0, bio_complex["msa"])

    l = len(new_order_list)
    perm_matrix = np.zeros((l, l), dtype=np.float32)
    perm_matrix[range(l), new_order_list] = 1.0

    for k in bio_complex:
        if "profile" in k:
            num_dim = bio_complex[k].shape.as_list()[-1]
            assert num_dim in [
                20,
                21,
                22,
            ], "num_dim for %s out of expected range: %s" % (k, num_dim)
            bio_complex[k] = torch.dot(bio_complex[k], perm_matrix[:num_dim, :num_dim])
    
    return bio_complex


def squeeze_features(bio_complex):
    """Remove singleton and repeated dimensions in bio_complex features."""
    if bio_complex["butype"].ndim == 2:
        # NOTE: data format capability
        bio_complex["butype"] = torch.argmax(bio_complex["butype"], dim=-1)
    if "template_butype" in bio_complex and bio_complex["template_butype"].ndim > 2:
            bio_complex["template_butype"] = torch.argmax(bio_complex["template_butype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in bio_complex:
            final_dim = bio_complex[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(bio_complex[k]):
                    bio_complex[k] = torch.squeeze(bio_complex[k], dim=-1)
                else:
                    bio_complex[k] = np.squeeze(bio_complex[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in bio_complex:
            bio_complex[k] = bio_complex[k][0]

    return bio_complex


@curry1
def randomly_replace_msa_with_unknown(bio_complex, replace_proportion):
    """Replace a portion of the MSA with 'X'."""
    msa_mask = torch.rand(bio_complex["msa"].shape) < replace_proportion
    x_idx = bio_complex["num_butype"]
    gap_idx = x_idx + 1
    msa_mask = torch.logical_and(msa_mask, bio_complex["msa"] != gap_idx)
    bio_complex["msa"] = torch.where(
        msa_mask,
        torch.ones_like(bio_complex["msa"]) * x_idx,
        bio_complex["msa"]
    )
    butype_mask = torch.rand(bio_complex["butype"].shape) < replace_proportion

    bio_complex["butype"] = torch.where(
        butype_mask,
        torch.ones_like(bio_complex["butype"]) * x_idx,
        bio_complex["butype"],
    )
    return bio_complex


@curry1
def sample_msa(bio_complex, max_seq, keep_extra, seed=None):
    """Sample MSA randomly, remaining sequences are stored are stored as `extra_*`.""" 
    num_seq = bio_complex["msa"].shape[0]
    g = torch.Generator(device=bio_complex["msa"].device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    shuffled = torch.randperm(num_seq - 1, generator=g) + 1
    index_order = torch.cat(
        (torch.tensor([0], device=shuffled.device), shuffled), 
        dim=0
    )

    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(
        index_order, [num_sel, num_seq - num_sel]
    )

    # NOTE: fix data bug
    if bio_complex['deletion_matrix'].shape != bio_complex['msa'].shape:
        bio_complex['deletion_matrix'] = padto(bio_complex['deletion_matrix'], bio_complex['msa'].shape)

    # TODO: RNA MSA sample?
    rna_pos = bio_complex["rna_pos"]
    for k in MSA_FEATURE_NAMES:
        if k in bio_complex:
            if keep_extra:
                bio_complex["extra_" + k] = torch.index_select(
                    bio_complex[k], 0, not_sel_seq
                )
            if len(rna_pos) > 0 and k != "msa_row_mask":
                rna_bak = bio_complex[k][:num_sel, rna_pos]
            bio_complex[k] = torch.index_select(bio_complex[k], 0, sel_seq)
            if len(rna_pos) > 0 and k != "msa_row_mask":
                rna_bak = bio_complex[k][:num_sel, rna_pos]
                bio_complex[k][:num_sel, rna_pos] = rna_bak

    return bio_complex


@curry1
def add_distillation_flag(bio_complex, distillation):
    bio_complex['is_distillation'] = distillation
    return bio_complex

@curry1
def sample_msa_distillation(bio_complex, max_seq):
    if(bio_complex["is_distillation"] == 1):
        bio_complex = sample_msa(max_seq, keep_extra=False)(bio_complex)
    return bio_complex


@curry1
def crop_extra_msa(bio_complex, max_extra_msa):
    num_seq = bio_complex["extra_msa"].shape[0]
    num_sel = min(max_extra_msa, num_seq)
    select_indices = torch.randperm(num_seq)[:num_sel]
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in bio_complex:
            bio_complex["extra_" + k] = torch.index_select(
                bio_complex["extra_" + k], 0, select_indices
            )
    
    return bio_complex


def delete_extra_msa(bio_complex):
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in bio_complex:
            del bio_complex["extra_" + k]
    return bio_complex


# Not used in inference
@curry1
def block_delete_msa(bio_complex, config):
    num_seq = bio_complex["msa"].shape[0]
    block_num_seq = torch.floor(
        torch.tensor(num_seq, dtype=torch.float32, device=bio_complex["msa"].device)
        * config.msa_fraction_per_block
    ).to(torch.int32)

    if config.randomize_num_blocks:
        nb = torch.distributions.uniform.Uniform(
            0, config.num_blocks + 1
        ).sample()
    else:
        nb = config.num_blocks

    del_block_starts = torch.distributions.Uniform(0, num_seq).sample(nb)
    del_blocks = del_block_starts[:, None] + torch.range(block_num_seq)
    del_blocks = torch.clip(del_blocks, 0, num_seq - 1)
    del_indices = torch.unique(torch.sort(torch.reshape(del_blocks, [-1])))[0]

    # Make sure we keep the original sequence
    combined = torch.cat((torch.range(1, num_seq)[None], del_indices[None]))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    keep_indices = torch.squeeze(difference, 0)

    for k in MSA_FEATURE_NAMES:
        if k in bio_complex:
            bio_complex[k] = torch.gather(bio_complex[k], keep_indices)

    return bio_complex


@curry1
def nearest_neighbor_clusters(bio_complex, gap_agreement_weight=0.0):
    num_butype = bio_complex["num_butype"]
    weights = torch.cat(
        [
            torch.ones(num_butype + 1, device=bio_complex["msa"].device), 
            gap_agreement_weight * torch.ones(1, device=bio_complex["msa"].device),
            torch.zeros(1, device=bio_complex["msa"].device)
        ],
        0,
    )

    # Make agreement score as weighted Hamming distance
    msa_one_hot = make_one_hot(bio_complex["msa"], num_butype + 3)
    sample_one_hot = bio_complex["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = make_one_hot(bio_complex["extra_msa"], num_butype + 3)
    extra_one_hot = bio_complex["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(
        torch.reshape(extra_one_hot, [extra_num_seq, num_res * (num_butype + 3)]),
        torch.reshape(
            sample_one_hot * weights, [num_seq, num_res * (num_butype + 3)]
        ).transpose(0, 1),
    )

    # Assign each sequence in the extra sequences to the closest MSA sample
    bio_complex["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).to(
        torch.int64
    )
    
    return bio_complex


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Similar to 
    tf.unsorted_segment_sum, but only supports 1-D indices.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The 1-D segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert (
        len(segment_ids.shape) == 1 and
        segment_ids.shape[0] == data.shape[0]
    )
    segment_ids = segment_ids.view(
        segment_ids.shape[0], *((1,) * len(data.shape[1:]))
    )
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = (
        torch.zeros(*shape, device=segment_ids.device)
        .scatter_add_(0, segment_ids, data.float())
    )
    tensor = tensor.type(data.dtype)
    return tensor


def summarize_clusters(bio_complex):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = bio_complex["msa"].shape[0]
    num_butype = bio_complex["num_butype"]

    def csum(x):
        return unsorted_segment_sum(
            x, bio_complex["extra_cluster_assignment"], num_seq
        )

    mask = bio_complex["extra_msa_mask"]
    mask_counts = 1e-6 + bio_complex["msa_mask"] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * make_one_hot(bio_complex["extra_msa"], num_butype + 3))
    msa_sum += make_one_hot(bio_complex["msa"], num_butype + 3)  # Original sequence
    bio_complex["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * bio_complex["extra_deletion_matrix"])
    del_sum += bio_complex["deletion_matrix"]  # Original sequence
    bio_complex["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum
    
    return bio_complex


def make_msa_mask(bio_complex):
    """Mask features are all ones, but will later be zero-padded."""
    bio_complex["msa_mask"] = torch.ones(bio_complex["msa"].shape, dtype=torch.float32)
    bio_complex["msa_row_mask"] = torch.ones(
        (bio_complex["msa"].shape[0]), dtype=torch.float32
    )
    return bio_complex


def pseudo_beta_fn(bio_complex, butype, all_atom_positions, all_atom_mask, complex_type=None):
    """Create pseudo beta features."""
    if "protein_pos" in bio_complex:
        protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    else:
        protein_pos, rna_pos = split_protein_rna_pos(bio_complex, complex_type)
    is_gly = torch.eq(butype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    o4_idx = nc.atom_order["O4'"]
    pseudo_beta = padcat(
        [
            torch.where(
                torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
                all_atom_positions[..., ca_idx, :],
                all_atom_positions[..., cb_idx, :],
            )[..., protein_pos, :],
            all_atom_positions[..., rna_pos, o4_idx, :]
        ],
        axis=-2
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = padcat(
            [
                torch.where(
                    is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
                )[..., protein_pos],
                all_atom_mask[..., rna_pos, o4_idx],
            ],
            axis=-1
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(bio_complex, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        bio_complex[prefix + "pseudo_beta"],
        bio_complex[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        bio_complex,
        bio_complex[prefix + "butype"],
        bio_complex[prefix + "all_atom_positions"],
        bio_complex[prefix + "all_atom_mask"],
    )
    return bio_complex


@curry1
def add_constant_field(bio_complex, key, value):
    bio_complex[key] = torch.tensor(value, device=bio_complex["msa"].device)
    return bio_complex


def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])


def make_hhblits_profile(bio_complex):
    """Compute the HHblits MSA profile if not already present."""
    if "hhblits_profile" in bio_complex:
        return bio_complex

    # Compute the profile for every residue (over all MSA sequences).
    msa_one_hot = make_one_hot(bio_complex["msa"], bio_complex["num_butype"] + 2) # 22, 6, 26

    bio_complex["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)
    return bio_complex


@curry1
def make_masked_msa(bio_complex, config, replace_fraction):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    num_butype = bio_complex["num_butype"]
    random_bu = torch.tensor(
        [0.05] * num_butype + [0.0, 0.0],
        dtype=torch.float32,
        device=bio_complex["butype"].device
    )

    categorical_probs = (
        config.uniform_prob * random_bu
        + config.profile_prob * bio_complex["hhblits_profile"]
        + config.same_prob * make_one_hot(bio_complex["msa"], num_butype + 2)
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(
        reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))])
    )
    pad_shapes[1] = 1
    mask_prob = (
        1.0 - config.profile_prob - config.same_prob - config.uniform_prob
    )
    assert mask_prob >= 0.0

    categorical_probs = torch.nn.functional.pad(
        categorical_probs, pad_shapes, value=mask_prob
    )

    sh = bio_complex["msa"].shape
    mask_position = torch.rand(sh) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, bio_complex["msa"])

    # Mix real and masked MSA
    bio_complex["bert_mask"] = mask_position.to(torch.float32)
    bio_complex["true_msa"] = bio_complex["msa"]
    bio_complex["msa"] = bert_msa

    return bio_complex


@curry1
def make_fixed_size(
    bio_complex,
    shape_schema,
    msa_cluster_size,
    extra_msa_size,
    num_res=0,
    num_templates=0,
):
    """Guess at the MSA and sequence dimension to make fixed size."""
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }
    if "origin_asym_id" in bio_complex:
        pad_size_map[NUM_FULL_RES] = max(bio_complex["origin_asym_id"].shape[0], num_res)

    for k, v in bio_complex.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            bio_complex[k] = torch.nn.functional.pad(v, padding)
            bio_complex[k] = torch.reshape(bio_complex[k], pad_size)
    
    return bio_complex


def make_msa_feat(bio_complex):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping for
    # compatibility with domain datasets.
    num_butype = bio_complex["num_butype"]
    butype_1hot = make_one_hot(bio_complex["butype"], num_butype + 1)
    if "between_segment_residues" in bio_complex:
        has_break = torch.clip(
            bio_complex["between_segment_residues"].to(torch.float32), 0, 1
        )
        target_feat = [
            torch.unsqueeze(has_break, dim=-1),
            butype_1hot,  # Everyone gets the original sequence.
        ]
    else:
        # NOTE: data format compability
        target_feat = [
            # NOTE: yujingcheng
            torch.zeros_like(bio_complex["butype"])[:, None],
            butype_1hot,  # Everyone gets the original sequence.
        ]

    msa_1hot = make_one_hot(bio_complex["msa"], num_butype + 3)
    has_deletion = torch.clip(bio_complex["deletion_matrix"], 0.0, 1.0)
    deletion_value = torch.atan(bio_complex["deletion_matrix"] / 3.0) * (
        2.0 / np.pi
    )

    msa_feat = [
        msa_1hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]

    if "cluster_profile" in bio_complex:
        deletion_mean_value = torch.atan(
            bio_complex["cluster_deletion_mean"] / 3.0
        ) * (2.0 / np.pi)
        msa_feat.extend(
            [
                bio_complex["cluster_profile"],
                torch.unsqueeze(deletion_mean_value, dim=-1),
            ]
        )

    if "extra_deletion_matrix" in bio_complex:
        bio_complex["extra_has_deletion"] = torch.clip(
            bio_complex["extra_deletion_matrix"], 0.0, 1.0
        )
        bio_complex["extra_deletion_value"] = torch.atan(
            bio_complex["extra_deletion_matrix"] / 3.0
        ) * (2.0 / np.pi)

    bio_complex["msa_feat"] = torch.cat(msa_feat, dim=-1)
    bio_complex["target_feat"] = torch.cat(target_feat, dim=-1)
    return bio_complex


@curry1
def select_feat(bio_complex, feature_list):
    return {k: v for k, v in bio_complex.items() if k in feature_list}


@curry1
def crop_templates(bio_complex, max_templates):
    for k, v in bio_complex.items():
        if k.startswith("template_"):
            bio_complex[k] = v[:max_templates]
    return bio_complex


@curry1
def make_dense_atom_masks(bio_complex, complex_type):
    device = bio_complex['butype'].device

    protein_map = rc.get_restype_atom_mapping(device)
    rna_map = nc.get_restype_atom_mapping(device)
    if complex_type == ComplexType.PROTEIN:
        mapping = protein_map
    elif complex_type == ComplexType.RNA:
        mapping = rna_map
    else:
        mapping = map_padcat(protein_map, rna_map)

    # Add mapping for unknown type
    padding = partial(torch.nn.functional.pad, pad=(0, 0, 0, 1))
    (
        restype_dense_to_all,
        restype_all_to_dense,
        restype_dense_mask,
        restype_all_mask,
    ) = map(padding, mapping)
    
    butype = bio_complex['butype'].to(torch.long)

    # create the mapping for (residx, atom_dense) --> all, i.e. an array
    # with shape (num_res, dense) containing the all indices for this bio_complex
    residx_dense_to_all = restype_dense_to_all[butype]
    residx_dense_mask = restype_dense_mask[butype]

    bio_complex["dense_atom_exists"] = residx_dense_mask
    bio_complex["residx_dense_to_all"] = residx_dense_to_all.long()

    # create the gather indices for mapping back
    residx_all_to_dense = restype_all_to_dense[butype]
    bio_complex["residx_all_to_dense"] = residx_all_to_dense.long()

    residx_all_mask = restype_all_mask[butype]
    bio_complex["all_atom_exists"] = residx_all_mask

    return bio_complex

    
def make_dense_atom_positions(bio_complex):
    residx_dense_mask = bio_complex["dense_atom_exists"]
    residx_dense_to_all = bio_complex["residx_dense_to_all"]

    # Create a mask for known ground truth positions.
    residx_dense_gt_mask = residx_dense_mask * batched_gather(
        bio_complex["all_atom_mask"],
        residx_dense_to_all,
        dim=-1,
        no_batch_dims=len(bio_complex["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_dense_gt_positions = residx_dense_gt_mask[..., None] * (
        batched_gather(
            bio_complex["all_atom_positions"],
            residx_dense_to_all,
            dim=-2,
            no_batch_dims=len(bio_complex["all_atom_positions"].shape[:-2]),
        )
    )

    bio_complex["dense_atom_exists"] = residx_dense_mask
    bio_complex["dense_atom_gt_exists"] = residx_dense_gt_mask
    bio_complex["dense_atom_gt_positions"] = residx_dense_gt_positions

    bio_complex = compute_alt_gt(bio_complex)

    return bio_complex

def compute_alt_gt(bio_complex):
    protein_pos = bio_complex["protein_pos"]
    butype = torch.clamp(bio_complex["butype"][protein_pos], max=20)

    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=bio_complex["all_atom_mask"].dtype,
            device=bio_complex["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            14, device=bio_complex["all_atom_mask"].device
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = rc.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = bio_complex["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[butype]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc", bio_complex["dense_atom_gt_positions"][protein_pos, :14], renaming_transform
    )
    bio_complex["dense_atom_alt_gt_positions"] = padto(
        alternative_gt_positions,
        bio_complex["dense_atom_gt_positions"].shape
    )

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb", bio_complex["dense_atom_gt_exists"][protein_pos, :14], renaming_transform
    )
    bio_complex["dense_atom_alt_gt_exists"] = padto(alternative_gt_mask, bio_complex["dense_atom_gt_exists"].shape)

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = bio_complex["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom14_names[resname].index(
                atom_name1
            )
            atom_idx2 = rc.restype_name_to_atom14_names[resname].index(
                atom_name2
            )
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    bio_complex["dense_atom_is_ambiguous"] = padto(
        restype_atom14_is_ambiguous[butype],
        bio_complex["dense_atom_gt_exists"].shape
    )
    return bio_complex


def atom37_to_frames(butype, all_atom_positions, all_atom_mask, eps=1e-8):
    batch_dims = len(butype.shape[:-1])
    butype = torch.clamp(butype, max=20)

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*butype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        rc.chi_angles_mask
    )

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = butype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        butype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        butype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=butype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=butype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        butype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        butype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(
        rot_mats=residx_rigidgroup_ambiguity_rot
    )
    alt_gt_frames = gt_frames.compose(
        Rigid(residx_rigidgroup_ambiguity_rot, None)
    )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    return (
        gt_frames_tensor,
        gt_exists,
        group_exists,
        residx_rigidgroup_is_ambiguous,
        alt_gt_frames_tensor
    )


def atom27_to_frames(butype, all_atom_positions, all_atom_mask, eps=1e-8):
    butype = correct_rna_butype(butype)

    batch_dims = len(butype.shape[:-1])
    # 5 NT types (AUGC plus X), 9 groups
    nttype_rigidgroup_base_atom_names = np.full([5, 9, 3], "", dtype=object)
    # atoms that constitute backbone frame 1
    nttype_rigidgroup_base_atom_names[:, 0, :] = ["O4'", "C4'", "C3'"]
    nttype_rigidgroup_base_atom_names[:, 1, :] = ["O4'", "C1'", "C2'"]

    for restype, restype_letter in enumerate(nc.restypes):
        # keep one-letter format in RNA
        resname = restype_letter
        for torsion_idx in range(7):
            if nc.chi_angles_mask[restype][torsion_idx]:
                names = nc.chi_angles_atoms[resname][torsion_idx]
                nttype_rigidgroup_base_atom_names[
                    restype, torsion_idx + 2, :
                ] = names[1:]
                
    # Or can be initiazed in all_ones for previous 4 dims as there are no missing frames in RNA
    nttype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*butype.shape[:-1], 5, 9),
    )
    nttype_rigidgroup_mask[..., 0] = 1
    nttype_rigidgroup_mask[..., 1] = 1
    nttype_rigidgroup_mask[..., :4, 2:] = all_atom_mask.new_tensor(
        nc.chi_angles_mask
    )

    lookuptable = nc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    # get atom index in atom_types that defined in nucleotide_constants
    nttype_rigidgroup_base_atom27_idx = lookup(
        nttype_rigidgroup_base_atom_names,
    )
    # 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    nttype_rigidgroup_base_atom27_idx = butype.new_tensor(
        nttype_rigidgroup_base_atom27_idx,
    )
    # # 1 * 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    nttype_rigidgroup_base_atom27_idx = (
        nttype_rigidgroup_base_atom27_idx.view(
            *((1,) * batch_dims), *nttype_rigidgroup_base_atom27_idx.shape
        )
    )
    # # N * 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    ntidx_rigidgroup_base_atom27_idx = batched_gather(
        nttype_rigidgroup_base_atom27_idx,
        butype.to(torch.long),
        dim=-3,
        no_batch_dims=batch_dims,
    )
    base_atom_pos = batched_gather(
        all_atom_positions,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )
    # # 0, 1, 2 are the index of frame atoms
    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=1e-8,
    )

    group_exists = batched_gather(
        nttype_rigidgroup_mask,
        butype.type(torch.long),
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists
    
    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=butype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 9, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)
    
    gt_frames = gt_frames.compose(Rigid(rots, None))
    
    gt_frames_tensor = gt_frames.to_tensor_4x4()

    residx_rigidgroup_is_ambiguous = torch.zeros_like(group_exists)
    alt_gt_frames_tensor = torch.zeros_like(gt_frames_tensor)

    return (
        gt_frames_tensor,
        gt_exists,
        group_exists,
        residx_rigidgroup_is_ambiguous,
        alt_gt_frames_tensor
    )

@curry1
def all_atom_to_frames(bio_complex, eps=1e-8):
    butype = bio_complex["butype"]
    all_atom_positions = bio_complex["all_atom_positions"]
    all_atom_mask = bio_complex["all_atom_mask"]
    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]

    (
        bio_complex["rigidgroups_gt_frames"],
        bio_complex["rigidgroups_gt_exists"],
        bio_complex["rigidgroups_group_exists"],
        bio_complex["rigidgroups_group_is_ambiguous"],
        bio_complex["rigidgroups_alt_gt_frames"],
    ) = map_padcat(
        atom37_to_frames(butype[protein_pos], all_atom_positions[protein_pos], all_atom_mask[protein_pos], eps),
        atom27_to_frames(butype[rna_pos], all_atom_positions[rna_pos], all_atom_mask[rna_pos], eps),
    )

    return bio_complex
    

def get_chi_atom_indices(complex_type):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.butypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    if complex_type == "protein":
        for residue_name in rc.restypes:
            residue_name = rc.restype_1to3[residue_name]
            residue_chi_angles = rc.chi_angles_atoms[residue_name]
            atom_indices = []
            for chi_angle in residue_chi_angles:
                atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
            for _ in range(4 - len(atom_indices)):
                atom_indices.append(
                    [0, 0, 0, 0]
                )  # For chi angles not defined on the AA.
            chi_atom_indices.append(atom_indices)

        chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    else:
        for residue_name in nc.restypes:
            residue_chi_angles = nc.chi_angles_atoms[residue_name]
            atom_indices = []
            for chi_angle in residue_chi_angles:
                atom_indices.append([nc.atom_order[atom] for atom in chi_angle])
            for _ in range(7 - len(atom_indices)):
                atom_indices.append(
                    [0, 0, 0, 0]
                )  # For chi angles not defined on the NT.
            chi_atom_indices.append(atom_indices)

        chi_atom_indices.append([[0, 0, 0, 0]] * 7)  # For UNKNOWN residue.
    
    return chi_atom_indices


def atom37_to_torsion_angles(
    butype,
    all_atom_positions,
    all_atom_mask,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)butype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    if butype.shape[-1] == 0:
        return None, None, None

    butype = torch.clamp(butype, max=20)

    num_atoms = all_atom_mask.shape[-1]

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, num_atoms, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, num_atoms])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices("protein"), device=butype.device
    )

    atom_indices = chi_atom_indices[..., butype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[butype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[butype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*butype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )


    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def atom27_to_torsion_angles(
    butype,
    all_atom_positions,
    all_atom_mask,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 27, 3] atom positions (in atom27
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 27] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    butype = correct_rna_butype(butype)
    butype = torch.clamp(butype, max=4)

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices("RNA"), device=butype.device
    )

    atom_indices = chi_atom_indices[..., butype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(nc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[butype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask
    # In the order of delta. gamma, beta, alpha1, alpha2, tm, chi
    torsions_atom_pos = chis_atom_pos
       
    torsion_angles_mask = chis_mask

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )
    
    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom
    
    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    alt_torsion_angles_sin_cos = torch.zeros_like(torsion_angles_sin_cos)

    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask

@curry1
def all_atom_to_torsion_angles(
    bio_complex,
    prefix="",
):
    butype = bio_complex[prefix + 'butype']
    all_atom_positions = bio_complex[prefix + 'all_atom_positions']
    all_atom_mask = bio_complex[prefix + 'all_atom_mask']

    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    (
        bio_complex[prefix + "torsion_angles_sin_cos"],
        bio_complex[prefix + "alt_torsion_angles_sin_cos"],
        bio_complex[prefix + "torsion_angles_mask"]
    ) = map_padcat(
        atom37_to_torsion_angles(
            butype[..., protein_pos],
            all_atom_positions[..., protein_pos, :, :],
            all_atom_mask[..., protein_pos, :]
        ),
        atom27_to_torsion_angles(
            butype[..., rna_pos],
            all_atom_positions[..., rna_pos, :27, :],
            all_atom_mask[..., rna_pos, :27]
        ),
        axis=butype.ndim - 1,
    )

    return bio_complex


def get_backbone_frames(bio_complex):
    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    bio_complex["backbone_rigid_tensor"] = padcat(
        [
            bio_complex["rigidgroups_gt_frames"][protein_pos, 0:1, :, :],
            bio_complex["rigidgroups_gt_frames"][rna_pos, 0:2, :, :],
        ],
    )
    bio_complex["backbone_rigid_mask"] = padcat(
        [
            bio_complex["rigidgroups_gt_exists"][protein_pos, 0:1],
            bio_complex["rigidgroups_gt_exists"][rna_pos, 0:2],
        ]
    )
    
    return bio_complex


def get_chi_angles(bio_complex):
    dtype = bio_complex["all_atom_mask"].dtype
    bio_complex["chi_angles_sin_cos"] = bio_complex["torsion_angles_sin_cos"].to(dtype)
    bio_complex["chi_mask"] = bio_complex["torsion_angles_mask"].to(dtype)
    return bio_complex


def _randint(lower, upper, device, g):
    # inclusive
    return int(torch.randint(
            lower,
            upper + 1,
            (1,),
            device=device,
            generator=g,
    )[0])

@curry1
def crop_to_size(
    bio_complex,
    crop_size,
    max_templates,
    shape_schema,
    subsample_templates=False,
    seed=None,
    spatial_crop_ratio=0.,
):
    device = bio_complex["seq_length"].device
    # We want each ensemble to be cropped the same way
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    randint = partial(_randint, device=device, g=g)

    r = torch.rand((1,), generator=g, device=device)[0]
    if r > spatial_crop_ratio:
        res_slices = random_crop_to_size(bio_complex, crop_size, randint)
    else:
        res_slices = spatial_crop_to_size_multiple_chains(bio_complex, crop_size, randint)
        if res_slices is None:
            # fallback to random crop
            res_slices = random_crop_to_size(bio_complex, crop_size, randint)

    if "template_mask" in bio_complex:
        num_templates = bio_complex["template_mask"].shape[-1]
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    if subsample_templates:
        templates_crop_start = randint(0, num_templates)
        templates_select_indices = torch.randperm(
            num_templates, device=device, generator=g
        )
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(
        num_templates - templates_crop_start, max_templates
    )

    if "asym_id" in bio_complex:
        keep_origin_keys = [
            "all_atom_positions",
            "all_atom_mask",
            "asym_id",
            "sym_id",
            "entity_id",
        ]
    else:
        keep_origin_keys = []

    origins = {}

    for k in keep_origin_keys:
        if k in bio_complex:
            origins['origin_' + k] = bio_complex[k]
    
    for k, v in bio_complex.items():
        if k not in shape_schema or (
            "template" not in k and NUM_RES not in shape_schema[k]
        ):
            continue

        # randomly permute the templates before cropping them.
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith("template"):
                slices.append(list(range(templates_crop_start, templates_crop_start + num_templates_crop_size)))
            elif is_num_res:
                slices.append(res_slices)
            else:
                slices.append(list(range(0, dim)))
        for i, s in enumerate(slices):
            v = torch.index_select(v, i, torch.tensor(s, device=v.device, dtype=torch.int64))

        bio_complex[k] = v

    bio_complex["seq_length"] = bio_complex["seq_length"].new_tensor(len(res_slices))
    bio_complex.update(origins)

    return bio_complex


def random_crop_to_size(
    bio_complex,
    crop_size,
    randint,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    if "asym_id" in bio_complex:
        return random_crop_to_size_multiple_chains(bio_complex, crop_size, randint)

    seq_length = bio_complex["seq_length"]
    num_res_crop_size = min(int(seq_length), crop_size)
    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in bio_complex and bio_complex["use_clamped_fape"] == 1.:
        right_anchor = n
    else:
        x = randint(0, n)
        right_anchor = n - x

    num_res_crop_start = randint(0, right_anchor)
    return list(range(num_res_crop_start, num_res_crop_start + num_res_crop_size))


def spatial_crop_to_size_multiple_chains(
    bio_complex,
    crop_size,
    randint,
):
    """Spatial cropping to `crop_size`, or keep as is if shorter than that.
    Implements Alphafold Multimer Algorithm 2, but make all homo chains have same cropping area.
    """

    ca_idx = rc.atom_order["CA"]
    ca_positions = bio_complex['all_atom_positions'][:, ca_idx, :]
    ca_masks = bio_complex['all_atom_mask'][:, ca_idx]
    asym_id = bio_complex['asym_id']

    pdist = torch.pairwise_distance(ca_positions[..., None, :], ca_positions[..., None, :, :])
    inter_chain_mask = ~(asym_id[..., :, None] == asym_id[..., None, :])
    # find interface residue pair
    interface_residue = torch.logical_and(pdist <= 6.0, inter_chain_mask)

    # remove masked out CAs
    interface_residue = torch.logical_and(interface_residue, ca_masks)
    interface_residue = torch.any(interface_residue, dim=-1)
    interface_residue = torch.logical_and(interface_residue, ca_masks)

    # get interface residues
    interface_residue = torch.flatten(torch.nonzero(interface_residue))

    if interface_residue.shape[-1] == 0:
        return None

    selected_interface_residue = interface_residue[randint(0, interface_residue.shape[-1] - 1)]
    center_pos = ca_positions[selected_interface_residue]

    center_dist = torch.pairwise_distance(ca_positions, center_pos)
    inf = torch.tensor(1e6)
    # remove masked out CAs
    center_dist += (1 - ca_masks) * inf

    return torch.sort(torch.argsort(center_dist)[:crop_size])[0].tolist()

def random_crop_to_size_multiple_chains(
    bio_complex,
    crop_size,
    randint,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way

    asym_id = bio_complex['asym_id']
    entity_id = bio_complex['entity_id']

    unique_asym_ids = torch.unique(asym_id).tolist()
    num_chains = len(unique_asym_ids)


    chains = {}
    for i in unique_asym_ids:
        where = torch.nonzero(asym_id == i)
        start = int(torch.min(where))
        end = int(torch.max(where) + 1)
        len_chain = end - start

        chains[i] = {
            'start': start,
            'len': len_chain
        }

    # Decide selecting area of each chain. Homo chains should have same selecting areas 
    # Each chain should have at least one res selected
    remain_res = int(asym_id.shape[0])
    remain_chains = num_chains
    remain_crop = crop_size

    selections = []
    for _, chain in chains.items():
        remain_res -= chain['len']
        remain_chains -= 1

        max_select = min(chain['len'], (remain_crop - remain_chains))
        min_select = min(chain['len'], max(1, (remain_crop - remain_res)))
        select_len = randint(min_select, max_select)
        select_start_offset = randint(0, chain['len'] - select_len)

        start = chain['start']
        selections.append((
                start + select_start_offset,
                start + select_start_offset + select_len,
            ))
        remain_crop -= select_len

    res_slices = []
    for start, end in selections:
        res_slices.extend(list(range(start, end)))

    return res_slices
