#$ -S /bin/bash

#$ -q serial
#$ -l h_vmem=4G
#$ -N conda-install

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2

export CONDA_ENVS_PATH=$global_storage/conda/envs
export CONDA_PKGS_DIRS=$global_storage/conda/pkgs
export PIP_CACHE_DIR=$global_storage/conda/pip

conda_save_location=$global_storage/conda_environments/py3.8-coastal-segmentation

conda-env create -p $conda_save_location --file ./environment.yaml

if source activate $conda_save_location; then
    pip install -r requirements.txt
else
    echo "Could not activate the conda environment at $conda_save_location"
fi