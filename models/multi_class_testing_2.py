import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
import cv2
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sn
import argparse
import neptune.new as neptune
from neptune.new.types import File
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
import config


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
        h=20
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

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.
        :returns: TODO

        """
        img_path = os.path.join(self._image_dir, self.images[index])
        mask_path = os.path.join(self._mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = self.mask_to_class_rgb(mask).cpu().detach().numpy()


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

def check_accuracyold(model_name, loader, model, run, device='cpu'):
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

    model.eval()  # set model for evaluation
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = y.to(torch.int64)
            # ===========================================================================
            x = x.permute(0, 3, 1, 2)
            #x = x.permute(0, 2, 3, 1)
            # get pixel predictions from image tensor
            preds = model(x.float())
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

            iou_score += a
            f1_score += b
            precision_score += c
            recall_score += d

    iou_score /= dataset_size  # averaged score across all images in directory
    f1_score /= dataset_size
    precision_score /= dataset_size
    recall_score /= dataset_size

    run['metrics/train/iou_score'].log(iou_score)
    run['metrics/train/f1_score'].log(f1_score)
    run['metrics/train/precision'].log(precision_score)
    run['metrics/train/recall'].log(recall_score)

def check_accuracy(run, loader, model, device='cpu'):
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

    model.eval() # set model for evaluation
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = y.to(torch.int64)
            #x = x.permute(0,1,2,3) # ===========================================================================
            x = x.permute(0,3,2,1)
            preds = model(x.float().contiguous())  # get pixel predictions from image tensor
            preds = torch.argmax(preds, dim=1).unsqueeze(1).int()  # get maximum values of tensor along dimension 1


            #print(preds.shape, y.shape)
            tp, fp, fn, tn = smpmetrics.get_stats(preds, y, mode='multiclass', num_classes=6)  # get tp,fp,fn,tn from predictions

            # compute metric
            a = smpmetrics.iou_score(tp, fp, fn, tn, reduction="macro")
            b = smpmetrics.f1_score(tp,fp,fn,tn, reduction='macro')
            c = smpmetrics.precision(tp,fp,fn,tn, reduction='macro')
            d = smpmetrics.recall(tp,fp,fn,tn, reduction='macro')

            iou_score += a
            f1_score += b
            precision_score += c
            recall_score += d

    iou_score /= dataset_size  # averaged score across all images in directory
    f1_score /= dataset_size
    precision_score /= dataset_size
    recall_score /= dataset_size

    run['metrics/train/iou_score'].log(iou_score)
    run['metrics/train/f1_score'].log(f1_score)
    run['metrics/train/precision'].log(precision_score)
    run['metrics/train/recall'].log(recall_score)

    print('IOU Score: {} | F1 Score: {} | Precision Score: {} | Recall Score: {}'.format(iou_score, f1_score, precision_score, recall_score))
    model.train()

def train_baseline(loader, model, optimizer, loss_fn, scaler, device, run):
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

        #update loop
        loop.set_postfix(loss=loss.item())


def train_wavelet(loader, model, optimizer, loss_fn, scaler, device, run):
    """ Custom training loop for models

    :loader: dataloader object
    :model: model to train
    :optimizer: training optimizer
    :loss_fn: loss function
    :scaler: scaler object
    :returns:

    """
    loop = tqdm(loader)  # just a nice library to keep track of loops

    for batch_idx, (data, targets) in enumerate(loop):  # iterate through dataset
        data = data.to(device=device).float()
        targets = targets.to(device=device)
        targets = targets.long()
        data = data.permute(0, 3, 1, 2)  # correct shape for image
        targets = targets.squeeze(1)

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

def train_cpu(loader, model, optimizer, loss_fn, device, run, epoch, iswavelet):
    """TODO: Docstring for train_fn.
    :loader: TODO
    :model: TODO
    :optimizer: TODO
    :loss_fn: TODO
    :scaler: TODO
    :returns: TODO
    """
    loop = tqdm(loader)
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device).float()
        targets = targets.to(device=device).float()
        targets = targets.unsqueeze(1)
        #data = data.permute(0,3,2,1)  # correct shape for image# ===========================================================================
        targets = targets.to(torch.int64)
        
        if iswavelet == 'no':
            data = data.permute(0,3,1,2)
            #print('hereweare')

        optimizer.zero_grad()
        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 2000 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        run['metrics/train/loss'].log(loss.item())

        # update loop
    
        loop.set_postfix(loss=loss.item())

def get_files(img_dir):
    path_list = []
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir, filename)
        if os.path.isfile(f):
            path_list.append(f)

    return path_list

def save_predictions_as_imgs(loader,
                             model,
                             run,
                             folder,
                             model_name,
                             device='cpu',
                             is_validation=False,
                             ):
    """TODO: Docstring for save_predictions_as_imgs.
    :loader: TODO
    :model: TODO
    :folder: TODO
    :device: TODO
    :returns: TODO
    """

    # define scores to track
    y_pred = []
    y_true = []
    # y_pred_tensor = torch.Tensor().cuda()
    # y_true_tensor = torch.Tensor().cuda()
    
    colors = [(0, 0, 255/255), (225/255, 0, 225/255), (255/255, 0, 0), (255/255, 225/255, 225/255), (255/255, 255/255, 0)]
    dataset_size = len(loader.dataset)  # number of images in the dataloader
    f1_score = 0
    precision_score = 0
    recall_score = 0
    iou_score = 0

    confmat = ConfusionMatrix(num_classes=6, normalize='true')
    CLASSES = ['background', 'ocean', 'wetsand',
               'buildings', 'vegetation', 'drysand']
    val_or_test = ''

    if is_validation:
        val_or_test = 'val'
    else:
        val_or_test = 'test'

    model.eval()  # set model for evaluation
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = y.to(torch.int64)
            # ===========================================================================
            x = x.permute(0, 1, 3, 2)
            #x = x.permute(0, 2, 3, 1)

            # get pixel predictions from image tensor
            preds = model(x.float())
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

            iou_score += a
            f1_score += b
            precision_score += c
            recall_score += d

            y_pred.append(preds)
            y_true.append(y)

    iou_score /= dataset_size  # averaged score across all images in directory
    f1_score /= dataset_size
    precision_score /= dataset_size
    recall_score /= dataset_size

    run[f'metrics/{val_or_test}/iou_score'].log(iou_score)
    run[f'metrics/{val_or_test}/f1_score'].log(f1_score)
    run[f'metrics/{val_or_test}/precision'].log(precision_score)
    run[f'metrics/{val_or_test}/recall'].log(recall_score)

    # print(len(y_pred), len(y_true), y_pred[0].shape, y_true[0].shape)
    cmp = ListedColormap(colors=colors)
    y_true = torch.cat(y_true, dim=2)

    y_pred = torch.cat(y_pred, dim=2)
    # print(y_true.shape, y_pred.shape)

    fop = y_true.squeeze().cpu().numpy()
    fop2 = y_pred.squeeze().cpu().numpy()

    # print(fop.shape, fop2.shape)

    rescaled = (255.0 / fop.max() * (fop - fop.min())).astype(np.uint8)
    rescaled2 = (255.0 / fop2.max() * (fop2 - fop2.min())).astype(np.uint8)
    
    matplotlib.image.imsave(f'{folder}multiclass_{val_or_test}_gt.jpg', rescaled, cmap=cmp)
    matplotlib.image.imsave(f'{folder}multiclass_{val_or_test}_preds.jpg', rescaled2, cmap=cmp)

    xut = y_pred
    xutrue = y_true

    ax = plt.axes()

    confmat = ConfusionMatrix(num_classes=6, normalize='true')
    df_cm = confmat(xut.cpu(), xutrue.cpu())
    df_cm = pd.DataFrame(df_cm.numpy())

    sn.set(font_scale=0.7)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 11})  # font size

    ax.set_title('{} Confusion Matrix'.format(model_name))
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_yticklabels(CLASSES, rotation=20)
    plt.savefig(f'{folder}multiclass_{val_or_test}_heatmap.jpg',
                dpi=200, bbox_inches="tight")
    plt.close()

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
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--iswavelet', type=str, required=True)

    args = parser.parse_args()

    run = neptune.init(
        project="PhD-Research/coastal-segmentation",
        source_files=['./*.ipynb', './*.py'],
        api_token=config.NEPTUNE_API_TOKEN,
    )

    TRAIN_IMG_DIR = args.trainimgdir
    TRAIN_MASK_DIR = args.trainmaskdir
    TEST_IMG_DIR = args.testimgdir
    TEST_MASK_DIR = args.testmaskdir
    VAL_IMG_DIR = args.valimgdir
    VAL_MASK_DIR = args.valmaskdir
    IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/images/'
    VAL_IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/val_images/'
    MODEL_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/model/'

    loss = smp.losses.DiceLoss(mode='multiclass')
    loss.__name__ = 'Dice_loss'
    LOSS_STR = 'Dice Loss'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OPTIM_NAME = 'Adam'

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
    run['parameters/model/classes'].log(args.classes)
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
                A.PadIfNeeded(
                    min_height=args.minheight, 
                    min_width=args.minwidth, 
                    border_mode=4
                    ),
                A.Resize(args.minheight, args.minwidth),
            ]
    )

    train_transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=args.minheight, 
                min_width=args.minwidth, 
                border_mode=4
                ),
            A.Resize(args.minheight, args.minwidth),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
        ]
    )

    val_transform = A.Compose(  # validation image transforms
        [
            A.Resize(256, 256), ToTensorV2()
        ]
    )

    trainDL, testDL = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  TEST_IMG_DIR, TEST_MASK_DIR,
                                  args.batchsize, train_transform,
                                  test_transform, num_workers=args.numworkers, pin_memory=True)

    valDL, valDL2 = get_loaders(VAL_IMG_DIR, VAL_MASK_DIR,
                                VAL_IMG_DIR, VAL_MASK_DIR,
                                args.valbatchsize, val_transform,
                                val_transform, num_workers=args.numworkers, pin_memory=True)

    if args.model == 'unet':
        print('starting unet')
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=6,
            activation=None,
        ).to(DEVICE)
    elif args.model == 'wavelet-unet':
        print('starting wavelet-unet')
        model = UNET(in_channels=3, out_channels=args.classes).to(DEVICE)
    elif args.model == 'manet':
        print('starting manet')
        model = smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=6,
            activation=None,
        ).to(DEVICE)
    elif args.model == 'pspnet':
        print('starting pspnet')
        model = smp.PSPNet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=None,
        ).to(DEVICE)
    elif args.model == 'fpn':
        print('starting fpn')
        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=None,
        ).to(DEVICE)
    elif args.model == 'unetpp':
        print('starting unetpp')
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=None,
        ).to(DEVICE)
    
    # optimizer = optim.AdamW(params=model.parameters(),
    #                         lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
    if OPTIM_NAME == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)


    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    scheduler = StepLR(optimizer=optimizer,
                    step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(args.epochs):  # run training and accuracy functions and save model
        run['parameters/epochs'].log(epoch)
        if torch.cuda.is_available():
            if args.model == 'wavelet-unet':
                print('training wavelet')
                train_wavelet(trainDL, model, optimizer, loss, scaler, DEVICE, run)
            else:
                print('training baseline')
                train_baseline(trainDL, model, optimizer, loss, scaler, DEVICE, run)
        else:
            train_cpu(trainDL, model, optimizer, loss, DEVICE, run, epoch)
        
        check_accuracy(run, testDL, model, DEVICE,)
        # check_accuracy2(args.model, testDL, model, run, DEVICE)
        scheduler.step()

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '{}multiclass_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            run["model_checkpoints/epoch{}".format(epoch)].upload('{}multi_class_{}.pt'.format(MODEL_SAVE_DIR, args.model))
            # save_predictions_as_imgs(
            #     testDL,
            #     model,
            #     run,
            #     folder=IMG_SAVE_DIR,
            #     model_name=args.model,
            #     device=DEVICE,
            #     is_validation=False
            # )

            save_predictions_as_imgs(
                valDL2,
                model,
                run,
                folder=VAL_IMG_SAVE_DIR,
                model_name=args.model,
                device=DEVICE,
                is_validation=True,
            )

            binary_result_paths = get_files(IMG_SAVE_DIR)
            binary_val_paths = get_files(VAL_IMG_SAVE_DIR)

            for image_path in binary_result_paths:
                       run["train/results/epoch{}".format(epoch)].log(File(image_path))

            for image_path2 in binary_val_paths:
                       run["train/validation/epoch{}".format(epoch)].log(File(image_path2))


if __name__ == '__main__':
    main()