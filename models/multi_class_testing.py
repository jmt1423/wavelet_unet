from unicodedata import name
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sn
import numpy as np
from multi_scale_unet import UNET
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as tvtransforms
import time
import torch
import segmentation_models_pytorch as smp
import metrics as smpmetrics
import albumentations as A
from tqdm import tqdm
from meter import AverageValueMeter
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import albumentations as A
from torch.utils.data import DataLoader

import torchmetrics
from torchmetrics import ConfusionMatrix
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from PIL import Image
import numpy as np
import os
import argparse
from neptune.new.types import File
import neptune.new as neptune
import config

from torch.optim.lr_scheduler import StepLR


class LastLoss:
    def __init__(self, ll = 0):
         self._ll = ll
      
    # getter method
    def get_ll(self):
        return self._ll
      
    # setter method
    def set_ll(self, x):
        self._ll = x


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

        self.mapping = {(0, 0, 0): 0, # background class (black)
                        (0, 0, 255): 1,  # 0 = class 1
                        (225, 0, 225): 2,  # 1 = class 2
                        (255, 0, 0): 3,  # 2 = class 3
                        (255, 225, 225): 4, # 3 = class 4
                        (255, 255, 0): 5}  # 4 = class 5

    def __len__(self):
        """returns length of images
        :returns: TODO

        """
        return len(self.images)
    
    def mask_to_class_rgb(self, mask):
        #print('----mask->rgb----')
        h=20 # CHANGE THISS ==================================================================================
        w=722
        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        #print('unique values rgb    ', torch.unique(mask)) 
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3PSPNet
        #print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])
       
        return mask_out


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
    
    train_ds = Dataset(image_dir=train_dir,
                             mask_dir=train_mask_dir,
                             transform=train_transform)
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

def check_accuracy(MODEL_NAME, loader, model, device='cpu'):
    """ Custom method to calculate accuracy of testing data
    :loader: dataloader objects
    :model: model to test
    :device: cpu or gpu
    """

    # define scores to track
    f1_score = 0
    precision_score = 0
    recall_score = 0
    iou_score = 0
    dataset_size = len(loader.dataset)  # number of images in the dataloader
    y_pred = []
    y_true = []
    confmat = ConfusionMatrix(num_classes=6, normalize='true')
    CLASSES = ['background', 'ocean', 'wetsand', 'buildings', 'vegetation', 'drysand']

    model.eval()  # set model for evaluation
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = y.to(torch.int64)
            x = x.permute(0,1,2,3) # ===========================================================================
            #x = x.permute(0, 2, 3, 1)
            # get pixel predictions from image tensor
            preds = model(x.float().contiguous())
            # get maximum values of tensor along dimension 1
            preds = torch.argmax(preds, dim=1).unsqueeze(1).int()

            #print(preds.shape, y.shape)
            tp, fp, fn, tn = smpmetrics.get_stats(
                preds, y, mode='multiclass', num_classes=6)  # get tp,fp,fn,tn from predictions

            # compute metric
            a = smpmetrics.iou_score(tp, fp, fn, tn, reduction="macro")
            b = smpmetrics.f1_score(tp, fp, fn, tn, reduction='macro')
            c = smpmetrics.precision(tp, fp, fn, tn, reduction='macro')
            d = smpmetrics.recall(tp, fp, fn, tn, reduction='macro')

            y_pred.extend(preds)
            y_true.extend(y)

            iou_score += a
            f1_score += b
            precision_score += c
            recall_score += d
            
    xut = y_pred[0]
    xutrue=y_true[0]

    ax = plt.axes()

    confmat = ConfusionMatrix(num_classes=6, normalize='true')
    df_cm = confmat(xut.cpu(), xutrue.cpu())
    df_cm = pd.DataFrame(df_cm.numpy())

    sn.set(font_scale=0.9)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size

    ax.set_title('{} Confusion Matrix'.format(MODEL_NAME))
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_yticklabels(CLASSES, rotation=20)
    plt.savefig('./{}_heatmap.png'.format(MODEL_NAME),dpi=300, bbox_inches = "tight")
    plt.show()

    iou_score /= dataset_size  # averaged score across all images in directory
    f1_score /= dataset_size
    precision_score /= dataset_size
    recall_score /= dataset_size

    # plt.close()
    print('IOU Score: {} | F1 Score: {} | Precision Score: {} | Recall Score: {}'.format(
        iou_score, f1_score, precision_score, recall_score))

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
        data = data.permute(0,3,1,2)  # correct shape for image
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
        del loss, predictions

def train_fn(loader, model, optimizer, loss_fn, scaler, device, ll, run):
    """ Custom training loop for models

    :loader: dataloader object
    :model: model to train
    :optimizer: training optimizer
    :loss_fn: loss function
    :scaler: scaler object
    :returns:

    """
    loop = tqdm(loader)  # just a nice library to keep track of loops
    model = model.to(device)# ===========================================================================
    for batch_idx, (data, targets) in enumerate(loop):  # iterate through dataset
        data = data.to(device=device).float()
        targets = targets.to(device=device).float()
        targets = targets.unsqueeze(1)
        data = data.permute(0,3,2,1)  # correct shape for image# ===========================================================================
        targets = targets.to(torch.int64)

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
        ll.set_ll(loss.item())

        #update loop
        loop.set_postfix(loss=loss.item())

def get_files(img_dir):
    path_list = []
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir,filename)
        if os.path.isfile(f):
            path_list.append(f)
    
    return path_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, required=True)
    parser.add_argument('--valbatchsize', type=int, required=True)
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
    parser.add_argument('--trainimgdir', type=str, required=True)
    parser.add_argument('--trainmaskdir', type=str, required=True)
    parser.add_argument('--testimgdir', type=str, required=True)
    parser.add_argument('--testmaskdir', type=str, required=True)
    parser.add_argument('--valimgdir', type=str, required=True)
    parser.add_argument('--valmaskdir', type=str, required=True)
    parser.add_argument('--numworkers', type=int, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    last_loss = LastLoss()

    avail = torch.cuda.is_available() # just checking which devices are available for training
    devCnt = torch.cuda.device_count()
    devName = torch.cuda.get_device_name(0)
    print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))
    
    run = neptune.init(
        project="jmt1423/coastal-segmentation",
        source_files=['./*.ipynb', './*.py'],
        api_token=config.NEPTUNE_API_TOKEN,
    )

    TRAIN_IMG_DIR = args.trainimgdir
    TRAIN_MASK_DIR = args.trainmaskdir
    TEST_IMG_DIR = args.testimgdir
    TEST_MASK_DIR = args.testmaskdir
    VAL_IMG_DIR = args.valimgdir
    VAL_MASK_DIR = args.valmaskdir
    IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/experiments/{args.experiment}/images/'
    VAL_IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/experiments/{args.experiment}/val_images/'
    MODEL_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/experiments/{args.experiment}/model/'

    loss = smp.losses.DiceLoss(mode='binary')
    LOSS_STR = 'Dice Loss'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OPTIM_NAME = 'AdamW'


    # log parameters to neptune
    run['parameters/model/model_name'].log(args.model)
    run['parameters/model/encoder'].log(args.encoder)
    run['parameters/model/encoder_weights'].log(args.encoderweights)
    run['parameters/model/activation'].log(args.activation)
    run['parameters/model/batch_size'].log(args.batchsize)
    run['parameters/model/learning_rate'].log(args.lr)
    run['parameters/model/loss'].log(LOSS_STR)
    run['parameters/model/device'].log(DEVICE)
    run['parameters/model/imgheight'].log(args.minheight)
    run['parameters/model/imgwidth'].log(args.minwidth)
    run['parameters/model/numworkers'].log(args.numworkers)
    run['parameters/optimizer/optimizer_name'].log(OPTIM_NAME)
    run['parameters/optimizer/beta1'].log(args.beta1)
    run['parameters/optimizer/beta2'].log(args.beta2)
    run['parameters/optimizer/epsilon'].log(args.epsilon)
    run['parameters/optimizer/stepsize'].log(args.stepsize)
    run['parameters/optimizer/gamma'].log(args.gamma)

    """
    Training and testing image transforms using albumentation libraries
    """
    test_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=args.minheight, min_width=args.minwidth, border_mode=4),
            A.Resize(args.minheight, args.minwidth),ToTensorV2()
        ]
    )

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=args.minheight, min_width=args.minwidth, border_mode=4),
            A.Resize(args.minheight, args.minwidth),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),ToTensorV2()
        ]
    )

    val_transform = A.Compose(  # validation image transforms
        [A.Resize(256, 256),ToTensorV2()]
    )

    trainDL, testDL = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                            TEST_IMG_DIR, TEST_MASK_DIR,
                            args.batchsize, train_transform,
                            test_transform, num_workers=args.numworkers, pin_memory=True)
    
    valDL, valDL2 = get_loaders(VAL_IMG_DIR, VAL_MASK_DIR,
                            VAL_IMG_DIR, VAL_MASK_DIR,
                            args.valbatchsize, val_transform,
                            val_transform, num_workers=args.numworkers, pin_memory=True)

    # initialize model
    #model = smp.Unet(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    in_channels=3,
    #    classes=len(CLASSES),
    #    activation=ACTIVATION,
    #)

    wavelet_model = UNET(in_channels=3, out_channels=6).to(DEVICE)

    optimizer = optim.AdamW(params=wavelet_model.parameters(),
                            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
    
    scaler = torch.cuda.amp.GradScaler()

    scheduler = StepLR(optimizer=optimizer,
                    step_size=args.stepsize, gamma=args.gamma)
    
    for epoch in range(args.epochs):  # run training and accuracy functions and save model
        run['parameters/epochs'].log(epoch)
        train_fn(trainDL, wavelet_model, optimizer, loss, scaler, DEVICE, last_loss, run)
        #train_wavelet(trainDL, wavelet_model, optimizer, loss, scaler)
        check_accuracy(args.model, testDL, wavelet_model, run, DEVICE)
        scheduler.step()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': wavelet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss.get_ll(),
            }, '{}binary_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            run["model_checkpoints/epoch{}".format(epoch)].upload('{}multi_class_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            save_predictions_as_imgs(
                testDL,
                model,
                folder=IMG_SAVE_DIR,
                device=DEVICE,
            )
            

            save_predictions_as_imgs(
                valDL,
                model,
                folder=VAL_IMG_SAVE_DIR,
                device=DEVICE,
            )
        
            binary_result_paths = get_files(IMG_SAVE_DIR)
            binary_val_paths = get_files(VAL_IMG_SAVE_DIR)

            for image_path in binary_result_paths:
                        run["train/results/epoch{}".format(epoch)].log(File(image_path))
            
            for image_path2 in binary_val_paths:
                        run["train/validation/epoch{}".format(epoch)].log(File(image_path2))

if __name__ == '__main__':
    main()
