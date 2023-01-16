#!/bin/bash

source scripts/activate_conda_env.sh

python3 train_opencomplex.py \
    --config_preset "mix"                                        `# config presets defined in config.py` \
    --complex_type "mix"                                         `# protein, RNA or mix (protein-RNA complex)` \
    --train_data_dir example_data/mmcif                          `# ground truth directory` \
    --train_feature_dir example_data/features                    `# features of training sample` \
    --train_filter_path example_data/filters/complex_filter.txt  `# optinal filter of training sample` \
    --val_data_dir example_data/mmcif                            `# optional ground truth directory of validation sample` \
    --val_feature_dir example_data/features                      `# optional features of validation sample` \
    --val_filter_path example_data/filters/complex_filter.txt    `# optioanl filter of validation sample` \
    --output_dir output/                                         `# output directory of checkpoints` \
    --precision 32                                               `# bf16 has better speed but may slightly lower accuracy` \
    --gpus 2 \
    --replace_sampler_ddp=True \
    --train_epoch_len 100 \
    --max_epochs 10 \
    --seed 4242022                                               `# in multi-gpu settings, the seed must be specified` \
    --checkpoint_every_epoch \
    --log_lr \
    --wandb
