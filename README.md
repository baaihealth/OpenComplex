# OpenComplex
OpenComplex is an open-source platform for developing protein and RNA complex models.  
Based on DeepMind's [Alphafold 2](https://github.com/deepmind/alphafold) and AQ Laboratory's [OpenFold](https://github.com/aqlaboratory/openfold), OpenComplex support almost all features from Alphafold 2 and OpenFold, and introduces the following new features:
* Reimplemented Alphafold-Multimer models.
* RNA and protein-RNA complex models with high precision.
* Kernel fusion and optimization on >=Ampere GPUs, brings 16% 

![Figure 1. OpenComplex inference result of RNA and protein-RNA complex.]()

We will release training results and pretrained parameters soon.

## Installation (Linux)

All Python dependencies are specified in `environment.yml`. For producing sequence 
alignments, you'll also need `kalign`, the [HH-suite](https://github.com/soedinglab/hh-suite), 
and one of {`jackhmmer`, [MMseqs2](https://github.com/soedinglab/mmseqs2) (nightly build)} 
installed on on your system.
Finally, some download scripts require `aria2c` and `aws`.

For convenience, we provide a script that installs Miniconda locally, creates a 
`conda` virtual environment, installs all Python dependencies, and downloads
useful resources, including both sets of model parameters. Run:

```bash
scripts/install_third_party_dependencies.sh
```

To activate the environment, run:

```bash
source scripts/activate_conda_env.sh
```

To deactivate it, run:

```bash
source scripts/deactivate_conda_env.sh
```

With the environment active, compile CUDA kernels with

```bash
python3 setup.py install
```

To install the HH-suite to `/usr/bin`, run

```bash
# scripts/install_hh_suite.sh
```

## Usage

### Data preparation

To run feature generation pipeline from `.fasta` to `feature.pkl` on DeepMind's MSA and template database, run e.g.:
```bash
python ./scripts/extract_pkl_from_fas.py ./example_data/fasta/ ./example_data/features/
```
where `example_data` is the directory containing example fasta . If `jackhmmer`, 
`hhblits`, `hhsearch` and `kalign` are available at the default path of 
`/usr/bin`, their `binary_path` command-line arguments can be dropped.
If you've already computed alignments for the query, you have the option to 
skip the expensive alignment computation here with 
`--use_precomputed_alignments`.

### Inference

To run inference with OpenComplex parameters, run e.g.:

```bash
python3 run_pretrained_opencomplex.py \
    --features_dir example_data/features_dir \        # the same dataset directory as in the previous step.
    --target_list_file example_data/filter.txt \ # filter of target lists
    --output_dir /path/to/output/directory \          # output directory
    --use_gpu \                                       # use gpu inference
    --num_workers 8 \                                 # number of parallel processes
    --param_paths /path/to/ckpt \                     # ckpt path
    --config_presets "RNA"                            # config presets as in config.py
    --complex_type "RNA"                              # protein, RNA, or mix (protein-RNA complex)
    --skip_relaxation \                               # skip amber relaxation
    --overwrite \                                     # overwrite existing result
```

### Training

After generating the feature files with the steps in data preparation section, call the training script:

```bash
python3 train_opencomplex.py
    --config_preset "RNA"                         # config presets defined in config.py
    --complex_type "RNA"                          # protein, RNA or mix (protein-RNA complex)
    --train_data_dir example_data/mmcif_dir       # ground truth directory
    --train_feature_dir example_data/features_dir # features of training sample
    --train_filter_path example/filter.txt        # filter of training sample
    --val_data_dir example_data/mmcif_dir         # optional ground truth directory of validation sample
    --val_feature_dir example_data/features_dir   # optional features of validation sample
    --val_filter_path example_data/filter.txt     # filter of validation sample
    --output_dir /path/to/output                  # output directory of checkpoints
    --precision 32                                # use bf16 will have better training speed but slightly worse accuracy
    --gpus 8
    --replace_sampler_ddp=True \
    --seed 4242022 \                              # in multi-gpu settings, the seed must be specified
    --train_epoch_len 5000 \
    --max_epochs 10 \
    --batch_size 1 \
    --checkpoint_every_epoch \
    --resume_from_ckpt ckpt_dir/ \
    --log_lr
    --wandb
```

## Testing

To run unit tests, use

```bash
scripts/run_unit_tests.sh
```

The script is a thin wrapper around Python's `unittest` suite, and recognizes
`unittest` arguments. E.g., to run a specific test verbosely:

```bash
scripts/run_unit_tests.sh -v tests.test_model
```

Certain tests require that AlphaFold (v2.0.1) be installed in the same Python
environment. These run components of AlphaFold and OpenFold side by side and
ensure that output activations are adequately similar. For most modules, we
target a maximum pointwise difference of `1e-4`.

## License and Disclaimer

Copyright 2022 BAAI.

Extended from AlphaFold and OpenFold, OpenComplex is licensed under
the permissive Apache Licence, Version 2.0.

## Contributing

If you encounter problems using OpenComplex, feel free to create an issue! We also
welcome pull requests from the community.

