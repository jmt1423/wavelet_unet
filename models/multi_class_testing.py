from torch.optim.lr_scheduler import StepLR
import config
import neptune.new as neptune
from neptune.new.types import File
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
import torch
import torch.optim as optim
import os
import seaborn as sn
import numpy as np
from multi_scale_unet import UNET
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import metrics as smpmetrics
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap

from torchmetrics import ConfusionMatrix
matplotlib.use('Agg')


class LastLoss:
    def __init__(self, ll=0):
        self._ll = ll

    # getter method
    def get_ll(self):
        return self._ll

    # setter method
    def set_ll(self, x):
        self._ll = x


class Dataset(Dataset):
    """This method creates the dataset from given directories"""

    def __init__(self, is_validation, image_dir, mask_dir, transform=None):
        """initialize directories

        :image_dir: TODO
        :mask_dir: TODO
        :transform: TODO

        """
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self.images = os.listdir(image_dir)

        if is_validation:
            print('validation')
            self.mapping = {(0, 0, 0): 0,  # background class (black)
                            (0, 0, 255): 1,  # 0 = class 1
                            (128, 0, 128): 2,  # 1 = class 2
                            (255, 0, 0): 3,  # 2 = class 3
                            (255, 255, 255): 4,  # 3 = class 4
                            (255, 255, 0): 5}  # 4 = class 5
        else:
            print('not validation')
            self.mapping = {(0, 0, 0): 0,  # background class (black)
                            (0, 0, 255): 1,  # 0 = class 1
                            (225, 0, 225): 2,  # 1 = class 2
                            (255, 0, 0): 3,  # 2 = class 3
                            (255, 225, 225): 4,  # 3 = class 4
                            (255, 255, 0): 5}  # 4 = class 5

    def __len__(self):
        """returns length of images
        :returns: TODO

        """
        return len(self.images)

    def mask_to_class_rgb(self, mask):
        h = 256  # ========================================================================================================================
        w = 256  # ========================================================================================================================
        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)  # remove 1

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(
                k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

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
    is_validation2=False,
):
    """
    This method creates the dataloader objects for the training loops

    :train_dir: directory of training images
    :train_mask_dir: directory of training masks
    :val_dir: validation image directory

    :returns: training and validation dataloaders
    recall
    """

    train_ds = Dataset(is_validation2, image_dir=train_dir,
                       mask_dir=train_mask_dir,
                       transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Dataset(is_validation2,
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


def check_accuracy(model_name, loader, model, run, device='cpu'):
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
    # ===========================================================================
    model = model.to(device)
    for batch_idx, (data, targets) in enumerate(loop):  # iterate through dataset
        # data = data.to(device=device).float()
        # targets = targets.to(device=device).float()
        # targets = targets.unsqueeze(1)
        # data = data.permute(0,3,2,1)  # correct shape for image# ===========================================================================
        # targets = targets.to(torch.int64)

        # # forward
        # with torch.cuda.amp.autocast():
        #     predictions = model(data)
        #     loss = loss_fn(predictions, targets)

        # # backward
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # # loss_values.append(loss.item())
        # run['training/batch/loss'].log(loss)
        # ll.set_ll(loss.item())

        # #update loop
        # loop.set_postfix(loss=loss.item())

        # =============================================
        # ============ wavelet training loop ==========
        # =============================================
        data = data.to(device=device).float()
        targets = targets.to(device=device)
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

        run['training/batch/loss'].log(loss)
        ll.set_ll(loss.item())

        # update loop
        loop.set_postfix(loss=loss.item())
        #del loss, predictions


def get_files(img_dir):
    path_list = []
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir, filename)
        if os.path.isfile(f):
            path_list.append(f)

    return path_list


def save_predictions_as_imgs(loader,
                             model,
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
    colors = [(0, 0, 255/255), (225/255, 0, 225/255), (255/255, 0, 0), (255/255, 225/255, 225/255), (255/255, 255/255, 0)]
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
            x = x.permute(0, 3, 1, 2)
            #x = x.permute(0, 2, 3, 1)

            # get pixel predictions from image tensor
            preds = model(x.float().contiguous())
            # get maximum values of tensor along dimension 1
            preds = torch.argmax(preds, dim=1).unsqueeze(1).int()

            y_pred.append(preds)
            y_true.append(y)

    #m_order = [2,0,1,3]
    #y_true = [y_true[i] for i in m_order]
    cmp = ListedColormap(colors=colors)
    y_true = torch.cat(y_true, dim=2)

    y_pred = torch.cat(y_pred, dim=2)

    fop = y_true.squeeze().cpu().numpy()
    fop2 = y_pred.squeeze().cpu().numpy()
    
    # rescaled = (255.0 / fop.max() * (fop - fop.min())).astype(np.uint8)
    # rescaled2 = (255.0 / fop2.max() * (fop2 - fop2.min())).astype(np.uint8)

    #matplotlib.image.imsave(f"{folder}multiclass_{val_or_test}_gt.jpg", fop, cmap=cmp)
    matplotlib.image.imsave('{folder}multiclass_{val_or_test}_gt.jpg'.format(folder, val_or_test), fop, cmap=cmp)
    matplotlib.image.imsave('{folder}multiclass_{val_or_test}_preds.jpg'.format(folder, val_or_test), fop2, cmap=cmp)

    #matplotlib.image.imsave(f'{folder}multiclass_{val_or_test}_preds.jpg', fop2, cmap=cmp)
    #plt.close()

    xut = y_pred
    xutrue = y_true

    ax = plt.axes()

    confmat = ConfusionMatrix(num_classes=6, normalize='true')
    df_cm = confmat(xut.cpu(), xutrue.cpu())
    df_cm = pd.DataFrame(df_cm.numpy())

    sn.set(font_scale=0.9)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size

    ax.set_title('{} Confusion Matrix'.format(model_name))
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_yticklabels(CLASSES, rotation=20)
    plt.savefig('{folder}multiclass_{val_or_test}_heatmap.jpg'.format(folder, val_or_test),
                dpi=100, bbox_inches="tight")
    # plt.show()


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

    last_loss = LastLoss()

    # just checking which devices are available for training
    # avail = torch.cuda.is_available()
    # devCnt = torch.cuda.device_count()
    # devName = torch.cuda.get_device_name(0)
    # print("Available: " + str(avail) + ", Count: " +
    #       str(devCnt) + ", Name: " + str(devName))

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
    IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/images/'
    VAL_IMG_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/val_images/'
    MODEL_SAVE_DIR = f'/storage/hpc/27/thomann/model_results/coastal_segmentation/{args.model}/multiclass/experiments/{args.experiment}/model/'

    loss = smp.losses.DiceLoss(mode='multiclass')
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
            A.Resize(args.minheight, args.minwidth),
        ]
    )

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=args.minheight,
                          min_width=args.minwidth, border_mode=4),
            A.Resize(args.minheight, args.minwidth),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.25,
                        interpolation=0, always_apply=False, p=0.5),
            A.Emboss(alpha=(0.2, 0.5), strength=(
                0.2, 0.7), always_apply=False, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = A.Compose(  # validation image transforms
        [A.Resize(256, 256)]
    )

    trainDL, testDL = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  TEST_IMG_DIR, TEST_MASK_DIR,
                                  args.batchsize, train_transform,
                                  test_transform, num_workers=args.numworkers, pin_memory=True, is_validation2=False)

    valDL, valDL2 = get_loaders(VAL_IMG_DIR, VAL_MASK_DIR,
                                VAL_IMG_DIR, VAL_MASK_DIR,
                                args.valbatchsize, val_transform,
                                val_transform, num_workers=args.numworkers, pin_memory=True, is_validation2=True)

    # initialize model
    # model = smp.Unet(
    #    encoder_name=ENCODER,
    #    encoder_weights=ENCODER_WEIGHTS,
    #    in_channels=3,
    #    classes=len(CLASSES),
    #    activation=ACTIVATION,
    # )

    wavelet_model = UNET(in_channels=3, out_channels=6).to(DEVICE)

    optimizer = optim.AdamW(params=wavelet_model.parameters(),
                            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
    # optimizer = optim.AdamW(params=wavelet_model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler()

    scheduler = StepLR(optimizer=optimizer,
                    step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(args.epochs):  # run training and accuracy functions and save model
        run['parameters/epochs'].log(epoch)
        train_fn(trainDL, wavelet_model, optimizer,
                 loss, scaler, DEVICE, last_loss, run)
        check_accuracy(args.model, testDL, wavelet_model, run, DEVICE)
        scheduler.step()

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': wavelet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': last_loss.get_ll(),
            }, '{}multiclass_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            run["model_checkpoints/epoch{}".format(epoch)].upload('{}multi_class_{}.pt'.format(MODEL_SAVE_DIR, args.model))

            save_predictions_as_imgs(
                testDL,
                wavelet_model,
                folder=IMG_SAVE_DIR,
                model_name=args.model,
                device=DEVICE,
                is_validation=False
            )

            save_predictions_as_imgs(
                valDL2,
                wavelet_model,
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