import argparse
import os

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

def save_predictions_as_imgs(val_or_test, metrics, loader, model, run, folder, device='cpu',):
    """TODO: Docstring for save_predictions_as_imgs.
    :loader: TODO
    :model: TODO
    :folder: TODO
    :device: TODO
    :returns: TODO
    """
    model.eval()
    metrics_meters = {metric.__name__: AverageValueMeter()
                      for metric in metrics}

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.float().to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))

            for metric_fn in metrics:
                metric_value = metric_fn(preds, y).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)

            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

            preds = (preds > 0.5).float()
            # print('saving_imagessssss')
            torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
            torchvision.utils.save_image(y, f"{folder}{idx}.png")
    
    # log metrics into neptune
    run[f'metrics/{val_or_test}/iou_score'].log(metrics_logs['iou_score'])
    run[f'metrics/{val_or_test}/f1_score'].log(metrics_logs['fscore'])
    run[f'metrics/{val_or_test}/precision'].log(metrics_logs['precision'])
    run[f'metrics/{val_or_test}/recall'].log(metrics_logs['recall'])

    model.train()

def train_fn(loader, model, optimizer, loss_fn, scaler, device, run, ll):
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
        data = data.to(device=device).float()
        targets = targets.to(device=device)
        targets = targets.long()
        #data = data.permute(0,3,1,2)  # correct shape for image
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
        run['metrics/train/loss'].log(loss.item())

        # update loop
        loop.set_postfix(loss=loss.item())
        del loss, predictions
        # data = data.to(device).float()
        # targets = targets.to(device).unsqueeze(1).float()
        # #data = data.permute(0,1,3,2)

        # # forward
        # with torch.cuda.amp.autocast():
        #     predictions = model(data)
        #     loss = loss_fn(predictions, targets)

        # # backward
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # run['metrics/train/loss'].log(loss.item())
        # ll.set_ll(loss.item())

        # # update loop
    
        # loop.set_postfix(loss=loss.item())

def train_cpu(loader, model, optimizer, loss_fn, device, run, epoch):
    """TODO: Docstring for train_fn.
    :loader: TODO
    :model: TODO
    :optimizer: TODO
    :loss_fn: TODO
    :scaler: TODO
    :returns: TODO
    """
    loop = tqdm(loader, disable=True)
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device).float()
        targets = targets.to(device).unsqueeze(1).float()
        #data = data.permute(0,1,3,2)

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
    parser.add_argument('--classes', type=int, required=True)

    args = parser.parse_args()
    
    run = neptune.init(
        project="jmt1423/coastal-segmentation",
        source_files=['./*.ipynb', './*.py'],
        api_token=config.NEPTUNE_API_TOKEN,
    )

    loss = smp.losses.DiceLoss(mode='binary')
    LOSS_STR = 'Dice Loss'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_loss = LastLoss()

    OPTIM_NAME = 'AdamW'

    TRAIN_IMG_DIR = args.trainimgdir
    TRAIN_MASK_DIR = args.trainmaskdir
    TEST_IMG_DIR = args.testimgdir
    TEST_MASK_DIR = args.testmaskdir
    VAL_IMG_DIR = args.valimgdir
    VAL_MASK_DIR = args.valmaskdir
    IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/binary/experiments/{args.experiment}/images/'
    VAL_IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/binary/experiments/{args.experiment}/val_images/'
    MODEL_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/binary/experiments/{args.experiment}/model/'


    # log parameters to neptune
    run['parameters/model/model_name'].log(args.model)
    run['parameters/model/encoder'].log(args.encoder)
    run['parameters/model/encoder_weights'].log(args.encoderweights)
    run['parameters/model/activation'].log(args.activation)
    run['parameters/model/batch_size'].log(args.batchsize)
    run['parameters/model/learning_rate'].log(args.lr)
    run['parameters/model/loss'].log(LOSS_STR)
    run['parameters/model/classes'].log(args.classes)
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

    train_transform = A.Compose([
        A.PadIfNeeded(min_height=args.minheight, min_width=args.minwidth, border_mode=4),
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

    val_transform = A.Compose(  # validation image transforms
        [A.Resize(256, 256),ToTensorV2()]
    )

        # get the dataloaders
    trainDL, testDL = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                TEST_IMG_DIR, TEST_MASK_DIR,
                                args.batchsize, train_transform,
                                test_transform, num_workers=args.numworkers, pin_memory=True)
    
    valDL, valDL2 = get_loaders(VAL_IMG_DIR, VAL_MASK_DIR,
                                VAL_IMG_DIR, VAL_MASK_DIR,
                                args.valbatchsize, val_transform,
                                val_transform, num_workers=args.numworkers, pin_memory=True)

    if args.model in ['unet']:
        print('starting unet')
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=1,
            activation=args.activation,
        ).to(DEVICE)
    elif args.model in ['wavelet-unet']:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif args.model in ['manet']:
        print('starting manet')
        model = smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=1,
            activation=args.activation,
        ).to(DEVICE)
    elif args.model in ['pspnet']:
        print('starting pspnet')
        model = smp.PSPNet(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=1,
            activation=args.activation,
        ).to(DEVICE)
    elif args.model in ['fpn']:
        print('starting fpn')
        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=1,
            activation=args.activation,
        ).to(DEVICE)
    elif args.model in ['unetpp']:
        print('starting unetpp')
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoderweights,
            in_channels=3,
            classes=1,
            activation=args.activation,
        ).to(DEVICE)
    
    # model = nn.DataParallel(model)

    # define optimizer and learning rate
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
    ]

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    scheduler = StepLR(optimizer=optimizer,
                    step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(args.epochs):  # run training and accuracy functions and save model
        run['parameters/epochs'].log(epoch)
        if torch.cuda.is_available():
            train_fn(trainDL, model, optimizer, loss, scaler, DEVICE, run, last_loss)
        else:
            train_cpu(trainDL, model, optimizer, loss, DEVICE, run, epoch)
        check_accuracy(metrics, testDL, model, run, DEVICE)
        scheduler.step()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss.get_ll(),
            }, '{}binary_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            run["model_checkpoints/epoch{}".format(epoch)].upload('{}binary_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            save_predictions_as_imgs(
                'test',
                metrics,
                testDL,
                model,
                run,
                folder=IMG_SAVE_DIR,
                device=DEVICE,
            )

            save_predictions_as_imgs(
                'validation',
                metrics,
                valDL,
                model,
                run,
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