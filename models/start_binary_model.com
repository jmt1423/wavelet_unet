#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=1
#$ -l h_vmem=60G
#$ -l h_rt=01:00:00
#$ -N binary-model

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2v2

source activate $global_storage/conda_environments/py3.8-coastal-segmentation

NOW=$(date +%Y%m%d_%H%M%S)
MODEL="fpn"
EXPERIMENT="$NOW"
TRAIN_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_train/'
TRAIN_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_trainannot/'
TEST_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_test/'
TEST_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_testannot/'
VAL_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/bval/'
VAL_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/bvalannot/'

# CLASSES=1
# BATCHSIZE=4
# VAL_BATCHSIZE=1
# LR=0.001
# GAMMA=0.05
# STEPSIZE=1
# EPOCHS=150
# ACTIVATION='sigmoid'
# ENCODER='resnet18'
# ENCODERWEIGHTS='imagenet'
# BETA1=0.7
# BETA2=0.99
# EPSILON=1e-7
# MINHEIGHT=32
# MINWIDTH=512
# NUMWORKERS=16

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
ACTIVATION="sigmoid"  # sigmoid, softmax
OPTIMIZER="AdamW_beta"  # Adam, AdamW, Adam_beta, AdamW_beta
BETA1=0.9
BETA2=0.99
EPSILON=1e-8

# learning rate parameters
SCHEDULER="steplr" # steplr, reducelronplateau
LR=0.001
SLR_GAMMA=0.0001
SLR_STEPSIZE=5
ROP_MODE="min"  # min, maximum
ROP_THRESHOLD_MODE="rel" # rel, abs
ROP_FACTOR=0.1
ROP_PATIENCE=10
ROP_THRESHOLD=1e-4
ROP_COOLDOWN=0
ROP_EPS=1e-8

# loss parameters
LOSS="Tversky"  # SCE, Dice, Tversky
REDUCTION="mean"
SMOOTH_FACTOR=0.1 
EPS_TVERSKY=1e-7
ALPHA_TVERSKY=0.5
BETA_TVERSKY=0.5
GAMMA_TVERSKY=1.0

# image parameters
MINHEIGHT=32
MINWIDTH=512

mkdir $global_storage/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT
mkdir $global_storage/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/model
mkdir $global_storage/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/images
mkdir $global_storage/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/val_images

python ./binary_testing.py --batchsize $BATCHSIZE --valbatchsize $VAL_BATCHSIZE --lr $LR --epochs $EPOCHS --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --minheight $MINHEIGHT --minwidth $MINWIDTH --slrgamma $SLR_GAMMA --slrstepsize $SLR_STEPSIZE --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR --testmaskdir $TEST_MASK_DIR --valimgdir $VAL_IMG_DIR --valmaskdir $VAL_MASK_DIR --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL --classes $CLASSES --optim $OPTIMIZER --loss $LOSS --epstversky $EPS_TVERSKY --alphatversky $ALPHA_TVERSKY --betatversky $BETA_TVERSKY --gammatversky $GAMMA_TVERSKY --scheduler $SCHEDULER --ropmode $ROP_MODE --roptmode $ROP_THRESHOLD_MODE --ropfactor $ROP_FACTOR --roppatience $ROP_PATIENCE --ropthreshold $ROP_THRESHOLD --ropcooldown $ROP_COOLDOWN --ropeps $ROP_EPS --scesmooth $SCE_SMOOTH_FACTOR --scereduction $SCE_REDUCTION