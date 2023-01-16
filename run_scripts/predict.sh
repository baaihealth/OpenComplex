#!/bin/bash
workdir=/sharefs/baaihealth/public_datasets/CASP/dataset0329/evalset_0412/
features_dir=$workdir/features
native_dir=$workdir/mmcif
target_list_file=$workdir/pdb_list.txt

output_dir=/tmp/predict_result

param_paths=/sharefs/baaihealth/yujingcheng/outputs/fast/opencomplex/e4sibfg4/checkpoints/14-18749.ckpt

gpu_num=`nvidia-smi --list-gpus | wc -l`
num_workers=$gpu_num
config_presets=fast

complex_type=protein

cmd="
python run_pretrained_opencomplex.py
    --features_dir $features_dir
    --target_list_file $target_list_file
    --output_dir $output_dir
    --use_gpu
    --config_presets $config_presets
    --param_paths $param_paths
    --save_outputs
    --num_workers $num_workers
    --native_dir $native_dir
    --skip_relaxation
    --overwrite
    --complex_type $complex_type
"
echo $cmd
eval $cmd
