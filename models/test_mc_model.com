#$ -S /bin/bash
#$ -q test
#$ -N mc_test

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2v2

source activate $global_storage/conda_environments/py3.8-coastal-segmentation

NOW=$(date +%Y%m%d_%H%M%S)
MODEL="wavelet-unet"
EXPERIMENT="$NOW"
TRAIN_IMG_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/train/"
TRAIN_MASK_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/trainannot/"
TEST_IMG_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/test/"
TEST_MASK_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/testannot/"
VAL_IMG_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/val2/"
VAL_MASK_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/valannot2/"

BATCHSIZE=1
CLASSES=5
VAL_BATCHSIZE=1
LR=0.000001
GAMMA=0.1
STEPSIZE=10
EPOCHS=200
ACTIVATION="ReLU"
ENCODER="resnet18"
ENCODERWEIGHTS="imagenet"
BETA1=0.9
BETA2=0.99
EPSILON=1e-8
MINHEIGHT=256
MINWIDTH=256
NUMWORKERS=16

mkdir /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT
mkdir /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/model
mkdir /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/images
mkdir /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/val_images

python ./multi_class_testing.py --batchsize $BATCHSIZE --valbatchsize $VAL_BATCHSIZE --lr $LR --epochs $EPOCHS --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --minheight $MINHEIGHT --minwidth $MINWIDTH --gamma $GAMMA --stepsize $STEPSIZE --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR --testmaskdir $TEST_MASK_DIR --valimgdir $VAL_IMG_DIR --valmaskdir $VAL_MASK_DIR --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL --classes $CLASSES