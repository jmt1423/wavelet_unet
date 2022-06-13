#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=2
#$ -l h_vmem=100G
#$ -l h_rt=11:00:00
#$ -N hyper-parameter-optim

source /etc/profile
module add anaconda3/wmlce
module add cuda/11.2v2

source activate $global_storage/conda_environments/py3.8-coastal-segmentation

NOW=$(date +%Y%m%d_%H%M%S)
MODEL="pspnet"
EXPERIMENT="$NOW"
TRAIN_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_train/'
TRAIN_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_trainannot/'
TEST_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_test/'
TEST_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/binary_testannot/'
VAL_IMG_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/bval/'
VAL_MASK_DIR='/storage/hpc/27/thomann/coastal_segmentation_data/current_data/bvalannot/'

#
# ─── IMAGE PARAMETERS ───────────────────────────────────────────────────────────
#

    
MINHEIGHT=256
MINWIDTH=256

#
# ─── PRETRAINED ENCODER WEIGHTS ─────────────────────────────────────────────────
#

    
ENCODER="resnet18"
ENCODERWEIGHTS="imagenet"

#
# ─── BASE PARAMETERS ────────────────────────────────────────────────────────────
#

    
CLASSES=1
VAL_BATCHSIZE=1
EPOCHS=50
NUMWORKERS=16
TRIALS=400

LOSS="Dice"  # SCE, Dice, Tversky
# Optimizer
ACTIVATION="sigmoid"  # sigmoid, softmax
OPTIMIZER="AdamW_beta"  # Adam, AdamW, Adam_beta, AdamW_beta
SCHEDULER="reducelronplataeu" # steplr, reducelronplataeu
#
# ─── SEARCH SPACE ───────────────────────────────────────────────────────────────
#

AUGMENT1="new"
AUGMENT2="small"

ROP_COOLDOWN=0
ROP_MODE="min"  # min, max
OPTIM_OBJECTIVE="f1" # recall, precision, f1, iou
MINMAX="maximize" # min, max

BATCHSIZE_LOW=5
BATCHSIZE_HIGH=20

LR_LOW=1e-4
LR_HIGH=1e-4

PATIENCE_LOW=1
PATIENCE_HIGH=15

OPTIM_EPS_LOW=1e-7
OPTIM_EPS_HIGH=1e-6

ROP_FACTOR_LOW=0.1
ROP_FACTOR_HIGH=0.5

ROP_EPS_LOW=1e-8
ROP_EPS_HIGH=1e-4

ROP_THRESH_LOW=1e-5
ROP_THRESH_HIGH=1e-1

BETA1_LOW=0.9
BETA1_HIGH=0.95

BETA2_LOW=0.95
BETA2_HIGH=0.99

LOSS_EPS_LOW=1e-8
LOSS_EPS_HIGH=1e-3

#
# ─── CROSS ENTROPY PARAMETERS ───────────────────────────────────────────────────
#

    
SCE_REDUCTION="mean"
SCE_SMOOTH_FACTOR=0.1
SLR_GAMMA=0.1
SLR_STEPSIZE=5

#
# ─── MAKE DIRECTORIES ───────────────────────────────────────────────────────────
#

    
mkdir -p /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT
mkdir -p /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/model
mkdir -p /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/images
mkdir -p /storage/hpc/27/thomann/model_results/coastal_segmentation/$MODEL/binary/experiments/$EXPERIMENT/val_images

#
# ─── ALL ARGUMENTS ──────────────────────────────────────────────────────────────
#

    
args=(
    --valbatchsize $VAL_BATCHSIZE --epochs $EPOCHS 
    --activation $ACTIVATION --encoder $ENCODER --encoderweights $ENCODERWEIGHTS 
    --minheight $MINHEIGHT 
    --minwidth $MINWIDTH --slrgamma $SLR_GAMMA --slrstepsize $SLR_STEPSIZE 
    --trainimgdir $TRAIN_IMG_DIR --trainmaskdir $TRAIN_MASK_DIR --testimgdir $TEST_IMG_DIR 
    --testmaskdir $TEST_MASK_DIR --valimgdir $VAL_IMG_DIR --valmaskdir $VAL_MASK_DIR 
    --numworkers $NUMWORKERS --experiment $EXPERIMENT --model $MODEL --classes $CLASSES --optim $OPTIMIZER --loss $LOSS 
    --scheduler $SCHEDULER --ropmode $ROP_MODE --ropcooldown $ROP_COOLDOWN --scesmooth $SCE_SMOOTH_FACTOR --scereduction $SCE_REDUCTION --trials $TRIALS --minmax $MINMAX --optimobjective $OPTIM_OBJECTIVE --lrlow $LR_LOW --lrhigh $LR_HIGH --patiencelow $PATIENCE_LOW --patiencehigh $PATIENCE_HIGH --optimepslow $OPTIM_EPS_LOW --optimepshigh $OPTIM_EPS_HIGH --ropfactorlow $ROP_FACTOR_LOW --ropfactorhigh $ROP_FACTOR_HIGH --ropepslow $ROP_EPS_LOW --ropepshigh $ROP_EPS_HIGH --ropthreshlow $ROP_THRESH_LOW --ropthreshhigh $ROP_THRESH_HIGH --beta1low $BETA1_LOW --beta1high $BETA1_HIGH --beta2low $BETA2_LOW --beta2high $BETA2_HIGH --lossepslow $LOSS_EPS_LOW --lossepshigh $LOSS_EPS_HIGH --batchsizelow $BATCHSIZE_LOW --batchsizehigh $BATCHSIZE_HIGH --augment1 $AUGMENT1 --augment2 $AUGMENT2
)

python ./hp_tuning_binary.py "${args[@]}"