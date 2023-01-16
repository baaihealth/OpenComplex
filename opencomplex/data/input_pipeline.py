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

from functools import partial

import torch

from opencomplex.data import data_transforms


def nonensembled_transform_fns(common_cfg, mode_cfg):
    """Input pipeline data transformers that are not ensembled."""
    complex_type = common_cfg.complex_type
    c_butype = common_cfg.c_butype
    unknown_type = common_cfg.get("unknown_type", -1)
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes(complex_type),
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(c_butype=c_butype, replace_proportion=0.0),
        data_transforms.make_seq_mask(unknown_type),
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile(c_butype),
    ]
    if common_cfg.use_templates:
        transforms.extend(
            [
                data_transforms.fix_templates_butype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta(prefix="template_", complex_type=complex_type),
            ]
        )
        if common_cfg.use_template_torsion_angles:
            transforms.extend(
                [
                    data_transforms.all_atom_to_torsion_angles(prefix="template_", complex_type=complex_type),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_dense_atom_masks(complex_type),
        ]
    )

    if mode_cfg.supervised:
        transforms.extend(
            [
                data_transforms.make_dense_atom_positions(complex_type),
                data_transforms.all_atom_to_frames(complex_type),
                data_transforms.all_atom_to_torsion_angles(prefix="", complex_type=complex_type),
                data_transforms.make_pseudo_beta(prefix="", complex_type=complex_type),
                data_transforms.get_backbone_frames(complex_type),
                data_transforms.get_chi_angles(complex_type),
            ]
        )

    return transforms


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if "max_distillation_msa_clusters" in mode_cfg:
        transforms.append(
            data_transforms.sample_msa_distillation(
                mode_cfg.max_distillation_msa_clusters
            )
        )

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if(not common_cfg.resample_msa_in_recycling):
        msa_seed = ensemble_seed
    
    transforms.append(
        data_transforms.sample_msa(
            max_msa_clusters, 
            keep_extra=True,
            seed=msa_seed,
        )
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                common_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction, c_butype=common_cfg.c_butype
            )
        )

    if common_cfg.msa_cluster_features:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat(c_butype=common_cfg.c_butype))

    crop_feats = dict(common_cfg.feat)

    if mode_cfg.fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        transforms.append(
            data_transforms.crop_to_size(
                mode_cfg.crop_size,
                mode_cfg.max_templates,
                crop_feats,
                mode_cfg.subsample_templates,
                seed=ensemble_seed + 1,
                spatial_crop_ratio = mode_cfg.get("spatial_crop_ratio", default=0.),
            )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                mode_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            )
        )
    else:
        transforms.append(
            data_transforms.crop_templates(mode_cfg.max_templates)
        )

    return transforms


def process_tensors_from_config(tensors, common_cfg, mode_cfg):
    """Based on the config, apply filters and transformations to the data."""
    ensemble_seed = torch.Generator().seed()

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg, 
            mode_cfg, 
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    no_templates = True
    if("template_butype" in tensors):
        no_templates = tensors["template_butype"].shape[0] == 0

    nonensembled = nonensembled_transform_fns(
        common_cfg,
        mode_cfg,
    )

    tensors = compose(nonensembled)(tensors)

    if("no_recycling_iters" in tensors):
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict
