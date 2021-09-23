from albumentations.augmentations.transforms import Lambda
from torch import optim
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.dataset import Dataset
from dataset import FlowerDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
from utils import (
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from tqdm import tqdm
from sklearn.metrics import f1_score
import neptune.new as neptune
from neptune.new.types import File

NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT')
API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

run = neptune.init(
    project=NEPTUNE_PROJECT,
    api_token=API_TOKEN,
    source_files=['*.py']
)

# training loop method
# TODO: can this be used with the original unet to? so i dont have to repeat it?
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """TODO: Docstring for train_fn.

    :loader: TODO
    :model: TODO
    :optimizer: TODO
    :loss_fn: TODO
    :scaler: TODO
    :returns: TODO

    """
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device='cuda')
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss_values.append(loss.item())
        run['training/batch/loss'].log(loss)

        #update loop
        loop.set_postfix(loss=loss.item())

TRAIN_IMG_DIR = "../drone_images/train_images/"
TRAIN_MASK_DIR = "../drone_images/train_masks_new/"
VAL_IMG_DIR = "../drone_images/val_images/"
VAL_MASK_DIR = "../drone_images/val_masks_new/"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 5
LEARNING_RATE = 1e-2
EPOCHS = 200

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
], )

val_transforms = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], )

# visualize ground truth and masks
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16,5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

dataset = FlowerDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
image, mask = dataset[4]
visualize(
    image=image,
    mask=mask.squeeze(),
)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['flower']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=1,
    activation=ACTIVATION,
)

preprocessing_fpn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                           VAL_IMG_DIR, VAL_MASK_DIR,
                                           BATCH_SIZE, train_transform,
                                           val_transforms)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall()
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# log parameters to neptune
run['parameters/epoch_nr'] = EPOCHS
run['parameters/batch_size'] = BATCH_SIZE
run['parameters/optimizer'] = 'Adam'
run['parameters/metrics'] = ['accuracy', 'dice_score', 'iou_score', 'precision', 'recall']
run['parameters/activation'] = ACTIVATION
run['parameters/encoder'] = ENCODER
run['parameters/image_height'] = IMAGE_HEIGHT
run['parameters/image_width'] = IMAGE_WIDTH
run['parameters/model'] = 'deeplabv3plus-2'
run['parameters/learning_rate'] = LEARNING_RATE


score = 0
scaler = torch.cuda.amp.GradScaler()

# IMPORTANT: Do not remove
# while they arent exactly used anywhere, they do put the encoder
# weights onto the cuda ready devices
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# custom training loop so i am able to manipulate tensors in the models
for epoch in range(EPOCHS):
    train_fn(train_loader, model, optimizer, loss, scaler)
    check_accuracy(metrics, run, val_loader, model, DEVICE)
    save_predictions_as_imgs(
        val_loader,
        model,
        folder='./saved_images/',
        device=DEVICE,
    )
