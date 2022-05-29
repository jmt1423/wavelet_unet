#$ -S /bin/bash
#$ -q short
#$ -l ngpus=2
#$ -l ncpus=5
#$ -l h_vmem=40G
#$ -l h_rt=01:00:00
#$ -N binary-model

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
VAL_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_val/'
VAL_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_valannot/'

CLASSES=1

BATCHSIZE=4
VAL_BATCHSIZE=1
LR=0.0001
GAMMA=0.1
STEPSIZE=5
EPOCHS=150
ACTIVATION='sigmoid'
ENCODER='resnet18'
ENCODERWEIGHTS='imagenet'
BETA1=0.95
BETA2=0.99
EPSILON=1e-7
MINHEIGHT=32
MINWIDTH=512
NUMWORKERS=15

mkdir $global_storage/model_results/coastal_segmentation/$model/binary/experiments/$EXPERIMENT
mkdir $global_storage/model_results/coastal_segmentation/$model/binary/experiments/$EXPERIMENT/model
mkdir $global_storage/model_results/coastal_segmentation/$model/binary/experiments/$EXPERIMENT/images
mkdir $global_storage/model_results/coastal_segmentation/$model/binary/experiments/$EXPERIMENT/val_images

python ./binary_testing.py --batchsize $BATCHSIZE --valbatchsize $VAL_BATCHSIZE --lr $LR --epochs $EPOCHS --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --minheight $MINHEIGHT --minwidth $MINWIDTH --gamma $GAMMA --stepsize $STEPSIZE --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR --testmaskdir $TEST_MASK_DIR --valimgdir $VAL_IMG_DIR --valmaskdir $VAL_MASK_DIR --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL --classes $CLASSES