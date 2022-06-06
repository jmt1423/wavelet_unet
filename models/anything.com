#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=3
#$ -l h_vmem=60G
#$ -l h_rt=01:00:00
#$ -N compute_class_weights

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2v2

source activate $global_storage/conda_environments/py3.8-coastal-segmentation

python ./class_weights.py