import argparse
import os
import time

import albumentations as A
import neptune.new as neptune
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from albumentations.pytorch import ToTensorV2
from neptune.new.types import File
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from meter import AverageValueMeter
from multi_scale_unet import UNET

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)

def main():
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
    parser.add_argument('--trainimgdir', type=str, required=True)
    parser.add_argument('--trainmaskdir', type=str, required=True)
    parser.add_argument('--testimgdir', type=str, required=True)
    parser.add_argument('--testmaskdir', type=str, required=True)
    parser.add_argument('--numworkers', type=int, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    avail = torch.cuda.is_available() # just checking which devices are available for training
    devCnt = torch.cuda.device_count()
    devName = torch.cuda.get_device_name(0)
    print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))
    
    run = neptune.init(
        project="jmt1423/coastal-segmentation",
        source_files=['./*.ipynb', './*.py'],
        api_token=config.NEPTUNE_API_TOKEN,
    )
    print('neptuneberunning')

    loss = smp.losses.DiceLoss(mode='binary')
    LOSS_STR = 'Dice Loss'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OPTIM_NAME = 'AdamW'

    TRAIN_IMG_DIR = args.trainimgdir
    TRAIN_MASK_DIR = args.trainmaskdir
    TEST_IMG_DIR = args.testimgdir
    TEST_MASK_DIR = args.testmaskdir
    IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/experiments/{args.experiment}/images/'
    MODEL_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/experiments/{args.experiment}/model/'


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
                                test_transform, num_workers=args.numworkers, pin_memory=True)
    
        # initialize model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoderweights,
        in_channels=3,
        classes=1,
        activation=args.activation,
    )
    model = model.to(DEVICE)
    model = nn.DataParallel(model)
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

    scaler = torch.cuda.amp.GradScaler()

    scheduler = StepLR(optimizer=optimizer,
                    step_size=args.stepsize, gamma=args.gamma)

    print(args.epochs)
    for epoch in range(args.epochs):  # run training and accuracy functions and save model
        run['parameters/epochs'].log(epoch)
        train_fn(trainDL, model, optimizer, loss, scaler, DEVICE, run)
        #train_wavelet(trainDL, wavelet_model, optimizer, loss, scaler)
        check_accuracy(metrics, testDL, model, run, DEVICE)
        scheduler.step()

    save_predictions_as_imgs(
            testDL,
            model,
            folder=IMG_SAVE_DIR,
            device=DEVICE,
        )
        
    binary_result_paths = get_files(IMG_SAVE_DIR)
    print('done training and saving')
    torch.save(model, '{}binary_{}.pth'.format(MODEL_SAVE_DIR, args.model))

    for image_path in binary_result_paths:
                run["train/results"].log(File(image_path))

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
    num_workers=5,
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

def check_accuracy(metrics, loader, model, run, device='cpu'):
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

    #print(metrics_logs)
    #print([type(k) for k in metrics_logs.values()])

    # log metrics into neptune
    run['metrics/train/iou_score'].log(metrics_logs['iou_score'])
    run['metrics/train/f1_score'].log(metrics_logs['fscore'])
    run['metrics/train/precision'].log(metrics_logs['precision'])
    run['metrics/train/recall'].log(metrics_logs['recall'])

    model.train()

def save_predictions_as_imgs(loader,
                             model,
                             folder,
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
        x = x.float()
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        print('saving_imagessssss')
        torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def train_fn(loader, model, optimizer, loss_fn, scaler, device, run):
    """TODO: Docstring for train_fn.
    :loader: TODO
    :model: TODO
    :optimizer: TODO
    :loss_fn: TODO
    :scaler: TODO
    :returns: TODO
    """
    loop = tqdm(loader, disable=True)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device).float()
        targets = targets.to(device).unsqueeze(1).float()
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

        #run['metrics/train/loss'].log(loss.item())
        # update loop
        loop.set_postfix(loss=loss.item())

def get_files(img_dir):
    path_list = []
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir,filename)
        if os.path.isfile(f):
            path_list.append(f)
    
    return path_list
            

if __name__ == '__main__':
    main()