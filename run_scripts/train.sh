#!/bin/bash
# Distribute Example
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export OMP_NUM_THREADS=4

if [ -z "$RLAUNCH_REPLICA_TOTAL" ]; then
	export RLAUNCH_REPLICA_TOTAL=1
fi

if [ -z "$RLAUNCH_REPLICA" ]; then
	export RLAUNCH_REPLICA=0
fi

if [ "$RLAUNCH_REPLICA" == "0" ]; then
	ifconfig $NCCL_SOCKET_IFNAME | grep inet | grep -v inet6 | awk '{print $2}' > $HOME/.master_ip
fi

function finish {
	rm -rf $HOME/.master_ip
}

trap finish EXIT INT TERM

while [ ! -f $HOME/.master_ip ]; do
	echo "wait $HOME/.master_ip..."
	ls > /dev/null && sleep 1;
done

export MASTER_ADDR=$(cat $HOME/.master_ip)
echo "master_ip: $MASTER_ADDR"

ulimit -n 40960

# =================== monomer dataset example ==========================
# workdir=/sharefs/baaihealth/public_datasets/CASP/dataset0225/
# mmcif_dir=$workdir/train/mmcif
# features_dir=$workdir/train/features
# val_mmcif_dir=$workdir/eval/mmcif
# val_features_dir=$workdir/eval/features
# train_chain_data_cache_path="--train_chain_data_cache_path $workdir/train/chain_data_cache.json"

# config_preset=fast

# complex_type=protein
# ======================================================================

# =================== multimer dataset example ==========================
# workdir=/sharefs/baaihealth/public_datasets/CASP/UltraFold/trainset/multimer_finetune
# mmcif_dir=$workdir/mmcif
# features_dir=$workdir/features
# val_mmcif_dir=$workdir/mmcif
# val_features_dir=$workdir/features

# config_preset=multimer_fast

# complex_type=protein
# ======================================================================

# =================== RNA dataset example ==========================
workdir=/sharefs/baaihealth/public_datasets/RNA/data_ready_all
mmcif_dir=$workdir/mmcif
features_dir=$workdir/feature
val_mmcif_dir=$workdir/mmcif
val_features_dir=$workdir/feature
train_label_dir="--train_label_dir $workdir/label"
val_label_dir="--val_label_dir $workdir/label"

config_preset=rna

complex_type=RNA
# ======================================================================


# in multi-gpu settings, the seed must be specified
seed=42


# Directory in which to output checkpoints
output_dir=/tmp/train_result

batch_size=1
train_epoch_len=5000
gpu_num=`nvidia-smi --list-gpus | wc -l`
max_epochs=15


# uncomment for debug mode, which sets data loader worker num to 1, and disable wandb
debug="--debug"


cmd="
MASTER_ADDR=$MASTER_ADDR MASTER_PORT=12345 WORLD_SIZE=$RLAUNCH_REPLICA_TOTAL NODE_RANK=$RLAUNCH_REPLICA
python train_opencomplex.py
    --train_data_dir ${mmcif_dir}
    --train_feature_dir ${features_dir}
    --val_data_dir ${val_mmcif_dir}
    --val_feature_dir ${val_features_dir}
    --output_dir ${output_dir}
    --precision 32
    --num_nodes 1
    --gpus ${gpu_num}
    --train_epoch_len ${train_epoch_len}
    --seed ${seed}
    --checkpoint_every_epoch
    --replace_sampler_ddp=True
    --config_preset ${config_preset}
    --log_lr
    --max_epochs ${max_epochs}
    --batch_size ${batch_size}
    --complex_type ${complex_type}
    --wandb
    $train_label_dir
    $val_label_dir
    ${train_chain_data_cache_path}
    ${debug}
"

echo $cmd
eval $cmd
