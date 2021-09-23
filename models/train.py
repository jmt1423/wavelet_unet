"""
This script trains the UNET model on the given dataset
"""
import albumentations as A
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp

from loss_functions import SoftDiceLoss
from multi_scale_unet import UNET
from baseline_unet import UNETablation
from utils import (check_accuracy, get_loaders, load_checkpoint,
                   save_predictions_as_imgs)
import neptune.new as neptune
from neptune.new.types import File

torch.cuda.empty_cache()

NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT')
API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

run = neptune.init(
    project=NEPTUNE_PROJECT,
    api_token=API_TOKEN,
    source_files=['*.py']
)

Image.MAX_IMAGE_PIXELS = 400000000

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MODEL_NAME = 'dwt_lowpass2_0f'
PIN_MEMORY = True
LOAD_MODEL = False  # set to true if you want to load the created checkpoint
TRAIN_IMG_DIR = "../drone_images/train_images/"
TRAIN_MASK_DIR = "../drone_images/train_masks_new/"
VAL_IMG_DIR = "../drone_images/val_images/"
VAL_MASK_DIR = "../drone_images/val_masks_new/"

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall()
]

# instantiate  neptune logging variables
run['parameters/epoch_nr'] = NUM_EPOCHS
run['parameters/batch_size'] = BATCH_SIZE
run['parameters/optimizer'] = 'Adam'
run['parameters/activation'] = 'ReLU'
run['parameters/encoder'] = 'se_resnext50_32x4d'
run['parameters/image_height'] = IMAGE_HEIGHT
run['parameters/image_width'] = IMAGE_WIDTH
run['parameters/model'] = MODEL_NAME
run['parameters/learning_rate'] = LEARNING_RATE

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
        data = data.to(device=DEVICE)
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
        run['training/batch/loss'].log(loss)

        # update loop
        loop.set_postfix(loss=loss.item())
        del loss, predictions


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
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
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], )

    val_transforms = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], )

    # to change to multiclass -> out_channels=num_of_classes
    # and change LogitsLoss to cross entropy loss
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = SoftDiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                           VAL_IMG_DIR, VAL_MASK_DIR,
                                           BATCH_SIZE, train_transform,
                                           val_transforms)

    if LOAD_MODEL:
        load_checkpoint(torch.load("./my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # check accuracy
        check_accuracy(metrics, run, val_loader, model, device=DEVICE)

        # print results to folder
        save_predictions_as_imgs(
            val_loader,
            model,
            folder="./saved_images/",
            device=DEVICE)


if __name__ == "__main__":
    main()
