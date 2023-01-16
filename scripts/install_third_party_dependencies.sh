#!/bin/bash
source scripts/vars.sh
$CONDA_PATH/bin/python3 -m pip install nvidia-pyindex
conda install -y mamba -c conda-forge -n base

conda create -y -n ${ENV_NAME} python=3.9
mamba env update --file environment.yml

source scripts/activate_conda_env.sh

echo "Attempting to install FlashAttention"
pip install git+https://github.com/HazyResearch/flash-attention.git@5b838a8bef78186196244a4156ec35bbb58c337d && echo "Installation successful"

# Install DeepMind's OpenMM patch
OPENCOMPLEX_DIR=$PWD
pushd $CONDA_PATH/envs/$ENV_NAME/lib/python3.9/site-packages/ \
    && patch -p0 < $OPENCOMPLEX_DIR/lib/openmm.patch \
    && popd

# Download folding resources
wget --no-check-certificate -P opencomplex/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Certain tests need access to this file
mkdir -p tests/test_data/alphafold/common
ln -rs opencomplex/resources/stereo_chemical_props.txt tests/test_data/alphafold/common
