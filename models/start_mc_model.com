#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=3
#$ -l h_vmem=60G
#$ -l h_rt=02:00:00
#$ -N multi-class-model

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
VAL_MASK_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/valannot3/"

# image parameters
MINHEIGHT=256
MINWIDTH=256

# pretrained encoder weights
ENCODER="resnet18"
ENCODERWEIGHTS="imagenet"

# Base parameters
BATCHSIZE=1
CLASSES=6
VAL_BATCHSIZE=1
EPOCHS=200
NUMWORKERS=4

# Optimizer
ACTIVATION="softmax"  # sigmoid, softmax
OPTIMIZER="AdamW_beta"  # Adam, AdamW, Adam_beta, AdamW_beta
BETA1=0.9
BETA2=0.99
EPSILON=1e-8

# learning rate parameters
SCHEDULER="reducelronplataeu" # steplr, reducelronplataeu
LR=0.0001
SLR_GAMMA=0.0000001
SLR_STEPSIZE=5
ROP_MODE="min"  # min, max
ROP_THRESHOLD_MODE="rel" # rel, abs
ROP_FACTOR=0.1
ROP_PATIENCE=10
ROP_THRESHOLD=1e-3
ROP_COOLDOWN=0
ROP_EPS=1e-8

# loss parameters
LOSS="Dice"  # SCE, Dice, Tversky
SCE_REDUCTION="mean"
SCE_SMOOTH_FACTOR=0.1
EPS_TVERSKY=1e-7
ALPHA_TVERSKY=0.5
BETA_TVERSKY=0.5
GAMMA_TVERSKY=1.0

mkdir -p $global_storage/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT
mkdir -p $global_storage/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/model
mkdir -p $global_storage/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/images
mkdir -p $global_storage/model_results/coastal_segmentation/$MODEL/multiclass/experiments/$EXPERIMENT/val_images

python ./hp_tuning.py --batchsize $BATCHSIZE --valbatchsize $VAL_BATCHSIZE --lr $LR --epochs $EPOCHS --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --minheight $MINHEIGHT --minwidth $MINWIDTH --slrgamma $SLR_GAMMA --slrstepsize $SLR_STEPSIZE --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR --testmaskdir $TEST_MASK_DIR --valimgdir $VAL_IMG_DIR --valmaskdir $VAL_MASK_DIR --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL --classes $CLASSES --optim $OPTIMIZER --loss $LOSS --epstversky $EPS_TVERSKY --alphatversky $ALPHA_TVERSKY --betatversky $BETA_TVERSKY --gammatversky $GAMMA_TVERSKY --scheduler $SCHEDULER --ropmode $ROP_MODE --roptmode $ROP_THRESHOLD_MODE --ropfactor $ROP_FACTOR --roppatience $ROP_PATIENCE --ropthreshold $ROP_THRESHOLD --ropcooldown $ROP_COOLDOWN --ropeps $ROP_EPS --scesmooth $SCE_SMOOTH_FACTOR --scereduction $SCE_REDUCTION