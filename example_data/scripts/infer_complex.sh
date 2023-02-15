#!/bin/bash

source scripts/activate_conda_env.sh

python3 run_pretrained_opencomplex.py \
    --features_dir example_data/features                        `# dir of generated features` \
    --target_list_file example_data/filters/complex_filter.txt  `# filter of target lists` \
    --output_dir output/infer_result                            `# output directory` \
    --use_gpu                                                   `# use gpu inference` \
    --num_workers 1                                             `# number of parallel processes` \
    --param_path /path/to/ckpt                                  `# ckpt path` \
    --config_preset "mix"                                       `# config presets as in config.py` \
    --complex_type "mix"                                        `# protein, RNA, or mix (protein-RNA complex)` \
    --skip_relaxation                                           `# skip amber relaxation` \
    --overwrite                                                 `# overwrite existing result`
