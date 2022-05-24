import argparse
import os
import time
from datetime import datetime

import albumentations as A
import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as tvtransforms
from albumentations.pytorch import ToTensorV2
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from neptune.new.types import File
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Dataset as BaseDataset
from torchmetrics import ConfusionMatrix
from tqdm import tqdm

import config
import metrics as smpmetrics
from meter import AverageValueMeter
from multi_scale_unet import UNET

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--activation', type=str, required=True)
parser.add_argument('--encoder', type=str, required=True)
parser.add_argument('--encoderweights', type=str, required=True)
parser.add_argument('--beta1', type=float, required=True)
parser.add_argument('--beta2', type=float, required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--minheight', type=int, required=True)
parser.add_argument('--minwidth', type=int, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--stepsize', type=int, required=True)

args = parser.parse_args()

time.sleep(40)
run = neptune.init(
    project="jmt1423/coastal-segmentation",
    source_files=['./*.ipynb', './*.py'],
    api_token=config.NEPTUNE_API_TOKEN,
)

# model parameters
#ENCODER = "resnet18"
#ENCODER_WEIGHTS = 'imagenet'
#ACTIVATION = 'sigmoid'
#BATCH_SIZE = 1
MODEL_NAME = 'unet'
#LEARNING_RATE = 0.00001
loss = smp.losses.DiceLoss(mode='binary')
LOSS_STR = 'Dice Loss'
#EPOCHS = 150
DEVICE = 'cuda'

# optimizer parameters
#BETA1 = 0.9
#BETA2 = 0.999
#EPSILON = 1e-8
OPTIM_NAME = 'AdamW'

# image parameters
#MIN_HEIGHT = 32
#MIN_WIDTH = 512

binary_result_paths = [
    'binary_results/0.png',
    'binary_results/1.png',
    'binary_results/2.png',
    'binary_results/3.png',
    'binary_results/4.png',
    'binary_results/5.png',
    'binary_results/6.png',
    'binary_results/7.png',
    'binary_results/8.png',
    'binary_results/9.png',
    'binary_results/10.png',
    'binary_results/11.png',
    'binary_results/12.png',
    'binary_results/13.png',
    'binary_results/14.png',
    'binary_results/15.png',
    'binary_results/pred_0.png',
    'binary_results/pred_1.png',
    'binary_results/pred_2.png',
    'binary_results/pred_3.png',
    'binary_results/pred_4.png',
    'binary_results/pred_5.png',
    'binary_results/pred_6.png',
    'binary_results/pred_7.png',
    'binary_results/pred_8.png',
    'binary_results/pred_9.png',
    'binary_results/pred_10.png',
    'binary_results/pred_11.png',
    'binary_results/pred_12.png',
    'binary_results/pred_13.png',
    'binary_results/pred_14.png',
    'binary_results/pred_15.png',
]

TRAIN_IMG_DIR = ""
TRAIN_MASK_DIR = ""
TEST_IMG_DIR = ""
TEST_MASK_DIR = ""
# VAL_IMG_DIR = ''
# VAL_MASK_DIR = ''


# log parameters to neptune
run['parameters/model/model_name'].log(MODEL_NAME)
run['parameters/model/encoder'].log(args.encoder)
run['parameters/model/encoder_weights'].log(args.encoderweights)
run['parameters/model/activation'].log(args.activation)
run['parameters/model/batch_size'].log(args.batchsize)
run['parameters/model/learning_rate'].log(args.lr)
run['parameters/model/loss'].log(LOSS_STR)
run['parameters/model/device'].log(DEVICE)
run['parameters/optimizer/optimizer_name'].log(OPTIM_NAME)
run['parameters/optimizer/beta1'].log(args.beta1)
run['parameters/optimizer/beta2'].log(args.beta2)
run['parameters/optimizer/epsilon'].log(args.epsilon)


class Dataset(Dataset):
    """This method creates the dataset from given directories"""

    def __init__(self, image_dir, mask_dir, transform=None):
        """initialize directories
        :image_dir: TODO
        :mask_dir: TODO
        :transform: TODO
        """
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """returns length of images
        :returns: TODO
        """
        return len(self.images)

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.
        :returns: TODO
        """
        img_path = os.path.join(self._image_dir, self.images[index])
        mask_path = os.path.join(self._mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1

        if self._transform is not None:
            augmentations = self._transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    This method creates the dataloader objects for the training loops

    :train_dir: directory of training images
    :train_mask_dir: directory of training masks
    :val_dir: validation image directory

    :returns: training and validation dataloaders
    recall
    """
    train_ds = Dataset(train_dir,
                       train_mask_dir,
                       transform=train_transform
                       )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Dataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


train_transform = A.Compose([
    A.Resize(height=args.minheight, width=args.minwidth),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                       rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                  hue=0.2, always_apply=False, p=0.5),
    A.Downscale(scale_min=0.25, scale_max=0.25,
                interpolation=0, always_apply=False, p=0.5),
    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
], )

test_transform = A.Compose([
    A.Resize(args.minheight, args.minwidth),
    ToTensorV2(),
], )

# get the dataloaders
trainDL, testDL = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                              TEST_IMG_DIR, TEST_MASK_DIR,
                              args.batchsize, train_transform,
                              test_transform)

# initialize model
model = smp.Unet(
    encoder_name=args.encoder,
    encoder_weights=args.encoderweights,
    in_channels=3,
    classes=1,
    activation=args.activation,
)

# wavelet_model = UNET(in_channels=3, out_channels=6).to(DEVICE)

# metrics have been defined in the custom training loop as giving them in a list object did not work for me


# define optimizer and learning rate
optimizer = optim.AdamW(params=model.parameters(),
                        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
]


def check_accuracy(metrics, loader, model, device='cpu'):
    """ Custom method to calculate accuracy of testing data
    :loader: dataloader objects
    :model: model to test
    :device: cpu or gpu
    """

    model.eval()  # set model for evaluation
    metrics_meters = {metric.__name__: AverageValueMeter()
                      for metric in metrics}
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).unsqueeze(1)
            #x = x.permute(0,1,3,2)
            preds = torch.sigmoid(model(x))

            for metric_fn in metrics:
                metric_value = metric_fn(preds, y).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)

            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

    print(metrics_logs)
    #print([type(k) for k in metrics_logs.values()])

    # log metrics into neptune
    run['metrics/train/iou_score'].log(metrics_logs['iou_score'])
    run['metrics/train/f1_score'].log(metrics_logs['fscore'])
    run['metrics/train/precision'].log(metrics_logs['precision'])
    run['metrics/train/recall'].log(metrics_logs['recall'])

    model.train()


def save_predictions_as_imgs(loader,
                             model,
                             folder="binary_results/",
                             device='cpu',
                             ):
    """TODO: Docstring for save_predictions_as_imgs.
    :loader: TODO
    :model: TODO
    :folder: TODO
    :device: TODO
    :returns: TODO
    """
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).float()
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


scaler = torch.cuda.amp.GradScaler()


def train_wavelet(loader, model, optimizer, loss_fn, scaler):
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
        data = data.to(device=DEVICE).float()
        targets = targets.to(device=DEVICE)
        targets = targets.long()
        data = data.permute(0, 3, 1, 2)  # correct shape for image
        targets = targets.squeeze(1)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data.contiguous())
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update loop
        loop.set_postfix(loss=loss.item())


scheduler = StepLR(optimizer=optimizer,
                   step_size=args.stepsize, gamma=args.gamma)

# def train_fn(loader, model, optimizer, loss_fn, scaler):
#     """ Custom training loop for models

#     :loader: dataloader object
#     :model: model to train
#     :optimizer: training optimizer
#     :loss_fn: loss function
#     :scaler: scaler object
#     :returns:

#     """
#     loop = tqdm(loader)  # just a nice library to keep track of loops
#     model = model.to(DEVICE)# ===========================================================================
#     for batch_idx, (data, targets) in enumerate(loop):  # iterate through dataset
#         data = data.to(device=DEVICE).float()
#         targets = targets.to(device=DEVICE).float()
#         targets = targets.unsqueeze(1)
#         data = data.permute(0,3,2,1)  # correct shape for image# ===========================================================================
#         targets = targets.to(torch.int64)

#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         # backward
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         # loss_values.append(loss.item())
#         #run['training/batch/loss'].log(loss)

#         #update loop
#         loop.set_postfix(loss=loss.item())


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
    model = model.to(DEVICE)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE).float()
        targets = targets.unsqueeze(1).to(device=DEVICE).float()
        #data = data.permute(0,1,3,2)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run['metrics/train/loss'].log(loss.item())
        # update loop
        loop.set_postfix(loss=loss.item())


for epoch in range(args.epochs):  # run training and accuracy functions and save model
    run['parameters/epochs'].log(epoch)
    train_fn(trainDL, model, optimizer, loss, scaler)
    #train_wavelet(trainDL, wavelet_model, optimizer, loss, scaler)
    check_accuracy(metrics, testDL, model, DEVICE)
    save_predictions_as_imgs(
        testDL,
        model,
        folder='./binary_results/',
        device=DEVICE,
    )
    torch.save(model, './binary_{}.pth'.format(MODEL_NAME))
    scheduler.step()

    # if epoch in {25, 50 ,75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400}:
    #     run["model_checkpoints/{}".format(MODEL_NAME)].upload("./binary_{}.pth".format(MODEL_NAME))

    # if epoch in {100, 200, 300, 400}:
    #     for image_path in binary_result_paths:
    #             run["train/results"].log(File(image_path))
print('doneeeeeeeeeeee')
