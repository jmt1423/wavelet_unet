#$ -S /bin/bash
#$ -q short
#$ -l ngpus=2
#$ -l ncpus=10
#$ -l h_vmem=60G
#$ -l h_rt=01:00:00
#$ -N run-coastal-model

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2v2

source activate $global_storage/conda_environments/py3.8-coastal-segmentation

NOW=$(date +%Y%m%d_%H%M%S)
MODEL="unet"
EXPERIMENT="$MODEL_$NOW"
TRAIN_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_train/'
TRAIN_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_trainannot/'
TEST_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_test/'
TEST_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_testannot/'

BATCHSIZE=4
LR=0.00000003
EPOCHS=100
ACTIVATION='sigmoid'
ENCODER='resnet18'
ENCODERWEIGHTS='imagenet'
BETA1=0.9
BETA2=0.99
EPSILON=1e-8
MINHEIGHT=32
MINWIDTH=512
GAMMA=0.1
STEPSIZE=20
NUMWORKERS=15

mkdir $global_storage/model_results/coastal_segmentation/unet/experiments/$EXPERIMENT
mkdir $global_storage/model_results/coastal_segmentation/unet/experiments/$EXPERIMENT/model
mkdir $global_storage/model_results/coastal_segmentation/unet/experiments/$EXPERIMENT/images

python ./binary_testing.py --batchsize $BATCHSIZE --lr $LR --epochs $EPOCHS --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --minheight $MINHEIGHT --minwidth $MINWIDTH --gamma $GAMMA --stepsize $STEPSIZE --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR --testmaskdir $TEST_MASK_DIR --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL