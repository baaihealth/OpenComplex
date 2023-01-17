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

import copy
from functools import partial
import json
import logging
import traceback
import os
import pickle
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import pytorch_lightning as pl
import torch

from opencomplex.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from opencomplex.utils.complex_utils import ComplexType
from opencomplex.utils.tensor_utils import tensor_tree_map, dict_multimap


class OpenComplexSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        config: mlc.ConfigDict,
        feature_dir: Optional[str] = None,
        complex_type: ComplexType = ComplexType.PROTEIN,
        filter_path: Optional[str] = None,
        mode: str = "train", 
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                config:
                    A dataset config object. See opencomplex.config
                feature_dir:
                    A path to a directory containing features.pkl
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(OpenComplexSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.config = config
        self.feature_dir = feature_dir
        self.mode = mode
        self.complex_type = complex_type

        self.supported_exts = [".cif"]

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        self._chain_ids = list(os.listdir(feature_dir))
        if(filter_path is not None):
            with open(filter_path, "r") as f:
                chains_to_include = set([l.strip() for l in f.readlines()])

            self._chain_ids = [c for c in self._chain_ids if c in chains_to_include]

        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=None,
        )

        self.feature_pipeline = feature_pipeline.FeaturePipeline(config) 
                
    def _parse_mmcif_with_prepared_feature(self, path, file_id, chain_id, feature_dir):
        if os.path.exists(os.path.join(feature_dir, 'mmcif_obj.pkl')):
            # Pre-compute mmcif_object for some of the samples to speedup computation
            mmcif_object = pickle.load(open(os.path.join(feature_dir, 'mmcif_obj.pkl'), 'rb'))
        else:
            with open(path, 'r') as f:
                mmcif_string = f.read()

            mmcif_object = mmcif_parsing.parse(
                file_id=file_id, mmcif_string=mmcif_string
            )

            # Crash if an error is encountered. Any parsing errors should have
            # been dealt with at the alignment stage.
            if(mmcif_object.mmcif_object is None):
                raise list(mmcif_object.errors.values())[0]

            mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif_with_prepared_feature(
            mmcif=mmcif_object,
            feature_dir=feature_dir,
            chain_id=chain_id,
            complex_type=self.complex_type,
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)
        feature_dir = os.path.join(self.feature_dir, name)

        if(self.mode == 'train' or self.mode == 'eval'):
            spl = os.path.splitext(name)[0].rsplit('_', 1)
            if (len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, chain_id = spl[0], None

            path = os.path.join(self.data_dir, file_id)
            ext = None
            for e in self.supported_exts:
                if(os.path.exists(path + e)):
                    ext = e
                    break

            if(ext is None):
                raise ValueError("Invalid file type")

            path += ext
            if(ext == ".cif"):
                data = self._parse_mmcif_with_prepared_feature(
                    path, file_id, chain_id, feature_dir
                )
            else:
               raise ValueError("Extension branch missing") 
        else:
            data = self.data_pipeline.process_prepared_features(feature_dir)

        feats = self.feature_pipeline.process_features(
            data, self.mode 
        )

        feats["batch_idx"] = torch.tensor([idx for _ in range(feats["butype"].shape[-1])],
                                          dtype=torch.int64, device=feats["butype"].device)

        return feats

    def __len__(self):
        return len(self._chain_ids) 


def deterministic_train_filter(
    chain_data_cache_entry: Any,
    max_resolution: float = 9.,
    max_single_aa_prop: float = 0.8,
) -> bool:
    # Hard filters
    resolution = chain_data_cache_entry.get("resolution", None)
    if(resolution is not None and resolution > max_resolution):
        return False

    seq = chain_data_cache_entry["seq"]
    counts = {}
    for aa in seq:
        counts.setdefault(aa, 0)
        counts[aa] += 1
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / len(seq)
    if(largest_single_aa_prop > max_single_aa_prop):
        return False

    return True


def get_stochastic_train_filter_prob(
    chain_data_cache_entry: Any,
) -> List[float]:
    # Stochastic filters
    probabilities = []
    
    cluster_size = chain_data_cache_entry.get("cluster_size", None)
    if(cluster_size is not None and cluster_size > 0):
        probabilities.append(1 / cluster_size)
    
    chain_length = len(chain_data_cache_entry["seq"])
    probabilities.append((1 / 512) * (max(min(chain_length, 512), 256)))

    # Risk of underflow here?
    out = 1
    for p in probabilities:
        out *= p

    return out


class OpenComplexDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenComplexFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """
    def __init__(self,
        datasets: Sequence[OpenComplexSingleDataset],
        probabilities: Sequence[int],
        epoch_len: int,
        chain_data_cache_paths: List[str],
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator
        
        self.chain_data_caches = []
        for path in chain_data_cache_paths:
            if path is None:
                continue
            with open(path, "r") as fp:
                self.chain_data_caches.append(json.load(fp))

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = int(epoch_len * probabilities[dataset_idx])
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))
            if self.chain_data_caches == []:
                chain_data_cache = None
            else:
                chain_data_cache = self.chain_data_caches[dataset_idx]
            while True:
                weights = []
                idx = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    chain_id = dataset.idx_to_chain_id(candidate_idx)
                    if chain_data_cache is None:
                        p = 1.0
                    else:
                        chain_data_cache_entry = chain_data_cache[chain_id]
                        if(not deterministic_train_filter(chain_data_cache_entry)):
                            continue

                        p = get_stochastic_train_filter_prob(
                            chain_data_cache_entry,
                        )
                    weights.append([1. - p, p])
                    idx.append(candidate_idx)

                samples = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=1,
                    generator=self.generator,
                )
                samples = samples.squeeze()
                
                if len(samples.shape) == 0:
                    samples = samples.unsqueeze(dim=0)

                cache = [i for i, s in zip(idx, samples) if s]

                for datapoint_idx in cache:
                    yield datapoint_idx

        self._samples = [looped_samples(i) for i in range(len(self.datasets))]

        if(_roll_at_init):
            self.reroll()

    def __getitem__(self, idx):
        try:
            dataset_idx, datapoint_idx = self.datapoints[idx]
            return self.datasets[dataset_idx][datapoint_idx]
        except Exception as e:
            # Manually handle exception to fix lightning bug
            traceback.print_exception(None, e, e.__traceback__)
            raise e

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class OpenComplexBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots) 


class OpenComplexDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage    

        if(generator is None):
            generator = torch.Generator()
        
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if(stage_cfg.supervised):
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            )
        
        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        butype = batch["butype"]
        batch_dims = butype.shape[:-2]
        recycling_dim = butype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample, 
                device=butype.device, 
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if(key == "no_recycling_iters"):
                no_recycling = sample 
        
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class OpenComplexDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        train_data_dir: Optional[str] = None,
        train_feature_dir: Optional[str] = None,
        train_chain_data_cache_path: Optional[str] = None,
        distillation_data_dir: Optional[str] = None,
        distillation_feature_dir: Optional[str] = None,
        distillation_alignment_dir: Optional[str] = None,
        distillation_chain_data_cache_path: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_feature_dir: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        predict_feature_dir: Optional[str] = None,
        train_filter_path: Optional[str] = None,
        val_filter_path: Optional[str] = None,
        distillation_filter_path: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: int = 50000, 
        complex_type: str = "protein",
        **kwargs
    ):
        super(OpenComplexDataModule, self).__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.train_feature_dir = train_feature_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.distillation_data_dir = distillation_data_dir
        self.distillation_feature_dir = distillation_feature_dir
        self.distillation_alignment_dir = distillation_alignment_dir
        self.distillation_chain_data_cache_path = (
            distillation_chain_data_cache_path
        )
        self.val_data_dir = val_data_dir
        self.val_feature_dir = val_feature_dir
        self.predict_data_dir = predict_data_dir
        self.predict_feature_dir = predict_feature_dir
        self.train_filter_path = train_filter_path
        self.val_filter_path = val_filter_path
        self.distillation_filter_path = distillation_filter_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len
        self.complex_type = ComplexType[complex_type]

        if(self.train_data_dir is None and self.predict_data_dir is None):
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None


    def setup(self, stage=None):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(OpenComplexSingleDataset,
            config=self.config,
            complex_type=self.complex_type,
        )

        if(self.training_mode):
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                feature_dir=self.train_feature_dir,
                filter_path=self.train_filter_path,
                mode="train",
            )

            distillation_dataset = None
            if(self.distillation_data_dir is not None):
                distillation_dataset = dataset_gen(
                    data_dir=self.distillation_data_dir,
                    feature_dir=self.distillation_feature_dir,
                    filter_path=self.distillation_filter_path,
                    mode="train",
                )

                d_prob = self.config.train.distillation_prob
           
            if(distillation_dataset is not None):
                datasets = [train_dataset, distillation_dataset]
                d_prob = self.config.train.distillation_prob
                probabilities = [1. - d_prob, d_prob]
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                    self.distillation_chain_data_cache_path,
                ]
            else:
                datasets = [train_dataset]
                probabilities = [1.]   
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                ]

            generator = None
            if(self.batch_seed is not None):
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)
            
            self.train_dataset = OpenComplexDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                chain_data_cache_paths=chain_data_cache_paths,
                generator=generator,
                _roll_at_init=False,
            )
    
            if(self.val_data_dir is not None):
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    feature_dir=self.val_feature_dir,
                    filter_path=self.val_filter_path,
                    mode="eval",
                )
            else:
                self.eval_dataset = None
        else:           
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                feature_dir=self.predict_feature_dir,
                filter_path=None,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = torch.Generator()
        if(self.batch_seed is not None):
            generator = generator.manual_seed(self.batch_seed)

        dataset = None
        if(stage == "train"):
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif(stage == "eval"):
            dataset = self.eval_dataset
        elif(stage == "predict"):
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenComplexBatchCollator()

        batch_size = self.config.data_module.data_loaders.batch_size
        if stage != "train" or self.complex_type == ComplexType.MIX:
            batch_size = 1

        dl = OpenComplexDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train") 

    def val_dataloader(self):
        if(self.eval_dataset is not None):
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict") 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(pl.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
