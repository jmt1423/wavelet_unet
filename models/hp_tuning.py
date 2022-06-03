"""
This file contains code for hyperparameter optimization using neptune, pytorch, and optuna.

Multiclass semantic segmentation

Models used:
 - unet
 - wavelet-unet (proposed model)
 - unet-plus-plus
 - fpn
 - pspnet
 - manet
 
 Classes:
 - 0: background
 - 1: ocean
 - 2: wetsand
 - 3 : buildings
 - 4 : vegetation
 - 5 : drysand

Software Citations:

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

https://github.com/qubvel/segmentation_models.pytorch
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import neptune.new as neptune
from neptune.new.types import File
import numpy as np
from multi_scale_unet import UNET
import torch
import optuna
import segmentation_models_pytorch as smp
import metrics as smpmetrics
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data.dataset import Dataset
import albumentations as A
from torch.utils.data import DataLoader
import neptune.new.integrations.optuna as optuna_utils

from PIL import Image
import numpy as np
import os
import config

torch.manual_seed(23)  # set random seed for reproducibility

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

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
parser.add_argument('--slrgamma', type=float, required=True)
parser.add_argument('--slrstepsize', type=int, required=True)
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
parser.add_argument('--optim', type=str, required=True)
parser.add_argument('--loss', type=str, required=True)
parser.add_argument('--scesmooth', type=float, required=True)
parser.add_argument('--scereduction', type=str, required=True)
parser.add_argument('--epstversky', type=float, required=True)
parser.add_argument('--alphatversky', type=float, required=True)
parser.add_argument('--betatversky', type=float, required=True)
parser.add_argument('--gammatversky', type=float, required=True)
parser.add_argument('--scheduler', type=str, required=True)
parser.add_argument('--ropmode', type=str, required=True)
parser.add_argument('--roptmode', type=str, required=True)
parser.add_argument('--ropfactor', type=float, required=True)
parser.add_argument('--roppatience', type=int, required=True)
parser.add_argument('--ropthreshold', type=float, required=True)
parser.add_argument('--ropcooldown', type=int, required=True)
parser.add_argument('--ropeps', type=float, required=True)
parser.add_argument('--trials', type=int, required=True)

args = parser.parse_args()

run = neptune.init(
    project="PhD-Research/coastal-segmentation",
    source_files=['./*.py'],
    api_token=config.NEPTUNE_API_TOKEN,
)

neptune_callback = optuna_utils.NeptuneCallback(run)


class Dataset(Dataset):
    """This method creates the dataset from given directories"""
    def __init__(self, iv, image_dir, mask_dir, transform=None):
        """initialize directories

        :image_dir: image directory
        :mask_dir: mask directory
        :transform: transforms to be applied to the images

        """
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self.images = os.listdir(image_dir)
        self.iv = iv

        self.mapping = {(0, 0, 0): 0, # background class (black)
                        (0, 0, 255): 1,  # 0 = class 1
                        (225, 0, 225): 2,  # 1 = class 2
                        (255, 0, 0): 3,  # 2 = class 3
                        (255, 225, 225): 4, # 3 = class 4
                        (255, 255, 0): 5}  # 4 = class 5

    def __len__(self):
        """
        :returns: length of images
        """
        return len(self.images)
    
    def mask_to_class_rgb(self, mask):
        if self.iv:  # validation images are different sizes so the dataset mask creation must change accordingly
            h=316
            w=316
        else:
            h=20
            w=722

        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)  # remove 1

        #print('unique values rgb    ', torch.unique(mask)) 

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

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
    iv,
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,):
    """
    This method creates the dataloader objects for the training loops

    :iv: boolean to determine if the dataset is validation or training
    :train_dir: training image directory
    :train_mask_dir: training mask directory
    :val_dir: validation image directory
    :val_mask_dir: validation mask directory
    :batch_size: batch size for training
    :train_transform: transforms to be applied to the training images
    :val_transform: transforms to be applied to the validation images
    :num_workers: number of workers for the dataloader
    :pin_memory: boolean to determine if the dataloader should use pin memory

    :returns: training and validation dataloaders
    """
    
    train_ds = Dataset(iv, image_dir=train_dir,
                             mask_dir=train_mask_dir,
                             transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Dataset(iv,
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

def check_accuracy(
    run, 
    loader, 
    model, 
    lossFn, 
    device='cpu', 
    is_validation=False
    ):
    """ 
    Custom method to calculate accuracy of testing data
    
    :run: neptune run object
    :loader: dataloader object
    :model: model object
    :lossFn: loss function object
    :device: device to use for the model
    :is_validation: boolean to determine if the dataloader is validation or training
    :returns: accuracy and loss of the model
    """

    # define scores to track
    f1_score = 0
    precision_score = 0
    recall_score = 0
    iou_score = 0
    dataset_size = len(loader.dataset)  # number of images in the dataloader
    y_pred = []
    y_true = []
    loss_total = 0
    val_or_test = ''

    if is_validation:
        val_or_test = 'val'
    else:
        val_or_test = 'test'

    model.eval() # set model for evaluation
    with torch.no_grad():  # do not calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = y.to(torch.int64)
            #x = x.permute(0,1,2,3) # ===========================================================================
            x = x.permute(0,1,2,3)
            preds = model(x.float().contiguous())  # get pixel predictions from image tensor
            
            loss = lossFn(preds, y)
            loss_total += loss.item()

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
    final_loss = loss_total/len(loader)

    run[f'metrics/{val_or_test}/iou_score'].log(iou_score)
    run[f'metrics/{val_or_test}/f1_score'].log(f1_score)
    run[f'metrics/{val_or_test}/precision'].log(precision_score)
    run[f'metrics/{val_or_test}/recall'].log(recall_score)

    model.train()

    return final_loss, iou_score, f1_score, precision_score, recall_score

def train_baseline(loader, model, optimizer, loss_fn, scaler, device, run):
    """ 
    Custom training loop for models

    :loader: dataloader object
    :model: model to train
    :optimizer: training optimizer
    :loss_fn: loss function
    :scaler: scaler object
    :device: device to use for the model
    :run: neptune run object
    
    :returns: None

    """
    loop = tqdm(loader, disable=True)  # just a nice library to keep track of loops
    # model = model.to(device)# ===========================================================================
    for batch_idx, (data, targets) in enumerate(loop):  # iterate through dataset
        data = data.to(device=device).float()
        targets = targets.to(device=device).float()
        targets = targets.unsqueeze(1)
        data = data.permute(0,3,1,2)  # correct shape for image# ===========================================================================
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
    """ 
    Custom training loop for models

    :loader: dataloader object
    :model: model to train
    :optimizer: training optimizer
    :loss_fn: loss function
    :scaler: scaler object
    :device: device to use for the data
    
    :returns: None
    """
    loop = tqdm(loader, disable=True)  # just a nice library to keep track of loops

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
    """
    Training loop for models on CPU
    
    :loader: dataloader object
    :model: model to train
    :optimizer: training optimizer
    :loss_fn: loss function
    :device: device to use for the data
    :run: neptune run object
    :epoch: current epoch
    :iswavelet: boolean for whether to use wavelet or not
    """
    loop = tqdm(loader, disable=True)
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

def get_loss(loss_name: str = "Dice", loss_eps: float = 1e-7):
    """
    get loss function
    
    :param loss_name: loss function name
    """
    if loss_name == 'Dice':
        loss = smp.losses.DiceLoss(mode='multiclass', eps=loss_eps)
    elif loss_name == 'SCE':
        loss = smp.losses.SoftCrossEntropyLoss(reduction=args.scereduction, smooth_factor=args.scesmooth)
        run['parameters/loss/reduction'].log(args.scereduction)
        run['parameters/loss/smooth_factor'].log(args.scesmooth)
    elif loss_name == 'Tversky':
        loss = smp.losses.TverskyLoss(
            mode='multiclass', 
            log_loss=False, 
            eps=args.epstversky, 
            alpha=args.alphatversky, 
            beta=args.betatversky, 
            gamma=args.gammatversky)
        run['parameters/loss/epsilon'].log(args.epstversky)
        run['parameters/loss/alpha'].log(args.alphatversky)
        run['parameters/loss/beta'].log(args.betatversky)
        run['parameters/loss/gamma'].log(args.gammatversky)
    return loss

def get_optimizer(model, 
                  optimizer_name: str = "AdamW_beta", 
                  optim_epsilon: float = 1e-8, 
                  lr: float = 1e-4,
                  beta1: float = 0.9,):
    """
    Get the optimizer for the model
    
    :param model: model to optimize
    :param optimizer_name: one of adamw_beta, adam_beta
    :param optim_epsilon: epsilon for adam
    :param lr: learning rate
    :param beta1: beta1 for adam
    """
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW_beta':
        optimizer = optim.AdamW(params=model.parameters(),
                             lr=lr, betas=(beta1, args.beta2), eps=optim_epsilon)
        run['parameters/optimizer/beta1'].log(beta1)
        run['parameters/optimizer/beta2'].log(args.beta2)
        run['parameters/optimizer/epsilon'].log(optim_epsilon)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
    elif optimizer_name == 'Adam_beta':
        optimizer = optim.Adam(params=model.parameters(),
                             lr=lr, betas=(beta1, args.beta2), eps=optim_epsilon)
        run['parameters/optimizer/beta1'].log(beta1)
        run['parameters/optimizer/beta2'].log(args.beta2)
        run['parameters/optimizer/epsilon'].log(optim_epsilon)

    run['parameters/optimizer/optimizer_name'].log(args.optim)
    return optimizer

def get_scheduler(optimizer, 
                  scheduler_name: str = "reducelronplataeu", 
                  patience: int = 10, 
                  rop_epsilon: float = 1e-8, 
                  rop_factor: float = 1,
                  rop_threshold: float = 1e-4,
                  rop_threshoold_mode: str = "rel"
                  ):
    """
    Get the scheduler for the optimizer
    
    :param optimizer: optimizer to use
    :param scheduler_name: one of reducelronplataeu, steplr
    :param patience: patience for reducelronplataeu
    :param rop_epsilon: epsilon for reducelronplataeu
    :param rop_factor: factor for reducelronplataeu
    :param rop_threshold: threshold for reducelronplataeu
    :param rop_threshoold_mode: mode for reducelronplataeu
    :returns: scheduler
    """
    
    if scheduler_name == 'steplr':
        scheduler = StepLR(optimizer=optimizer, step_size=args.slrstepsize, gamma=args.slrgamma)
        run['parameters/scheduler/stepsize'].log(args.slrstepsize)
        run['parameters/scheduler/gamma'].log(args.slrgamma)
    elif scheduler_name == 'reducelronplataeu':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, 
            mode=rop_threshoold_mode, 
            factor=rop_factor, 
            patience=patience, 
            threshold_mode=args.roptmode, 
            cooldown=args.ropcooldown, 
            threshold=rop_threshold,
            eps=rop_epsilon)
        run['parameters/scheduler/mode'].log(args.ropmode)
        run['parameters/scheduler/factor'].log(rop_factor)
        run['parameters/scheduler/threshold_mode'].log(rop_threshoold_mode)
        run['parameters/scheduler/cooldown'].log(args.ropcooldown)
        run['parameters/scheduler/threshold'].log(rop_threshold)
        run['parameters/scheduler/eps'].log(rop_epsilon)
    return scheduler

def get_model(model_name: str = "unet"):
    """
    Get mode for hyperparameter tuning
    
    :param model_name: str - name of model
    :return: model
    """
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=6,
            activation=args.activation,
        ).to(DEVICE)
    elif model_name == 'wavelet-unet':
        model = UNET(in_channels=3, out_channels=args.classes).to(DEVICE)
    elif model_name == 'manet':
        model = smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=6,
            activation=args.activation,
        ).to(DEVICE)
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=args.activation,
        ).to(DEVICE)
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=args.activation,
        ).to(DEVICE)
    elif model_name == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=args.classes,
            activation=args.activation,
        ).to(DEVICE)
    return model

def train_and_evaluate(model, optimizer, loss, scheduler):
    """
    Train and evaluate the model
    
    :param model: one of unet, manet, pspnet, fpn, unetpp, wavelet-unet
    :param optimizer: one of AdamW, Adam, AdamW_beta, Adam_beta
    :param loss: one of SCE, dice, tversky
    :param scheduler: one of steplr, reducelronplataeu
    :return: loss, iou, f1, precision, recall
    """
    
    TRAIN_IMG_DIR = args.trainimgdir
    TRAIN_MASK_DIR = args.trainmaskdir
    TEST_IMG_DIR = args.testimgdir
    TEST_MASK_DIR = args.testmaskdir
    VAL_IMG_DIR = args.valimgdir
    VAL_MASK_DIR = args.valmaskdir

    test_transform = A.Compose(        
            [
                A.Resize(args.minheight, args.minwidth),ToTensorV2(),
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

    # ─── DATA LOADERS ───────────────────────────────────────────────────────────────

    trainDL, testDL = get_loaders(False, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  TEST_IMG_DIR, TEST_MASK_DIR,
                                  args.batchsize, train_transform,
                                  test_transform, num_workers=args.numworkers, pin_memory=True)

    valDL, valDL2 = get_loaders(True, VAL_IMG_DIR, VAL_MASK_DIR,
                                VAL_IMG_DIR, VAL_MASK_DIR,
                                args.valbatchsize, val_transform,
                                val_transform, num_workers=args.numworkers, pin_memory=True)

    for epoch in range(args.epochs): 
        run['parameters/epochs'].log(epoch)
        if torch.cuda.is_available():
            if args.model == 'wavelet-unet':
                train_wavelet(trainDL, model, optimizer, loss, scaler, DEVICE, run)
            else:
                train_baseline(trainDL, model, optimizer, loss, scaler, DEVICE, run)
        else:
            train_cpu(trainDL, model, optimizer, loss, DEVICE, run, epoch)
        
        m, iou, f1, prec, rec = check_accuracy(run, valDL2, model, loss, DEVICE, is_validation=True)
            
        if args.scheduler == 'reducelronplataeu':
            scheduler.step(m)
        else: 
            scheduler.step()
        
    return m, iou, f1, prec, rec

def objective(trial):
    """
    Objective function for hyperparameter optimization.
    
    :param trial: A trial object from hyperopt.
    :return: loss, iou, f1, precision, recall
    """
    params = {
        # "model_name": trial.suggest_categorical('model_name',["wavelet-unet"]),
        # "loss_name": trial.suggest_categorical('loss_name',["SCE", "Dice", "Tversky"]),
        # "scheduler_name": trial.suggest_categorical('scheduler_name',["steplr", "reducelronplataeu"]),
        "lr": trial.suggest_float('lr', 1e-6, 1e-2),
        "optimizer_name": trial.suggest_categorical('optimizer_name',["Adam_beta", "AdamW_beta"]),
        "patience": trial.suggest_int("patience", 2, 10),
        "optim_epsilon": trial.suggest_float("optim_epsilon", 1e-8, 1e-2),
        "rop_epsilon": trial.suggest_float("rop_epsilon", 1e-8, 1e-3),
        "rop_factor": trial.suggest_float("rop_factor", 0.1, 0.6),
        "rop_threshold": trial.suggest_float("rop_threshold", 1e-6, 1e-2),
        "rop_threshold_mode": trial.suggest_categorical("rop_threshold_mode", ["abs", "rel"]),
        "beta1": trial.suggest_float("beta1", 0.6, 0.99),
    }

    model = get_model("wavelet-unet")

    optimizer = get_optimizer(model, 
                              optimizer_name=params["optimizer_name"],
                              optim_epsilon=params["optim_epsilon"], 
                              lr=params["lr"],
                              beta1=params["beta1"])
    scheduler = get_scheduler(optimizer, 
                              "reducelronplataeu", 
                              patience=params["patience"], 
                              rop_epsilon=params["rop_epsilon"], 
                              rop_factor=params["rop_factor"],
                              rop_threshold=params["rop_threshold"], 
                              rop_threshold_mode=params["rop_threshold_mode"])
    loss = get_loss("Dice")

    m, iou, f1, prec, rec = train_and_evaluate(model, optimizer, loss, scheduler)

    return m, iou, f1, prec, rec

def main():
    run['parameters/model/encoder'].log(args.encoder)
    run['parameters/model/encoder_weights'].log(args.encoderweights)
    run['parameters/model/activation'].log(args.activation)
    run['parameters/model/batch_size'].log(args.batchsize)
    run['parameters/model/device'].log(DEVICE)
    run['parameters/model/numworkers'].log(args.numworkers)
    run['parameters/model/classes'].log(args.classes)

    run['parameters/image/imgheight'].log(args.minheight)
    run['parameters/image/imgwidth'].log(args.minwidth)

    run['parameters/seed'].log(23)


    """
    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) l(x) 
    to the set of parameter values associated with the best objective values, and another 
    GMM g(x) to the remaining parameter values. It chooses the parameter value x that maximizes the ratio l(x)/g(x). 

    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
    """
    sampler = optuna.samplers.TPESampler()    
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=5, interval_steps=3
        ),directions=['minimize', 'maximize', 'maximize', 'maximize', 'maximize'])
    study.optimize(func=objective, n_trials=args.trials, callbacks=[neptune_callback])

if __name__ == '__main__':
    main()