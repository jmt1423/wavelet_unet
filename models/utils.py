import torch
import torchvision
from models.other.dataset import FlowerDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import re
import models.other.functional as F
import time
from models.meter import AverageValueMeter

DEVICE = 'cuda'


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Metric(BaseObject):
    pass

class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class IoU(Metric):
    __name__ = 'iou_score'
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class Recall(Metric):
    __name__ = 'recall_score'
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class Precision(Metric):
    __name__ = 'precision_score'
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """saves model at specified checkpoint/state

    :state: TODO
    :filename: TODO
    :returns: TODO

    """
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """loads model at specified checkpoint

    :checkpoint: TODO
    :model: TODO
    :returns: TODO

    """
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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
    """TODO: Docstring for get_loaders.

    :train_dir: TODO
    :train_mask_dir: TODO
    :val_dir: TODO
    :: TODO
    :returns: TODO

    """
    train_ds = FlowerDataset(image_dir=train_dir,
                             mask_dir=train_mask_dir,
                             transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FlowerDataset(
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

# TODO 
def check_accuracy(metrics, loader, model, device='cpu'):
    """TODO: Docstring for check_accuracy.
    :loader: TODO
    :model: TODO
    :device: TODO
    :returns: TODO
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()
            #num_correct += (preds == y).sum()
            #num_pixels += torch.numel(preds)
            #dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            for metric_fn in metrics:
                    metric_value = metric_fn(preds, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
    #accuracy = num_correct/num_pixels*100
    #print(
    #   f"Got {num_correct}/{num_pixels} with accuracy {accuracy:.2f}"
    #)
    print(metrics_logs)
    #dice_score = dice_score/len(loader)
    #run['validation/epoch/accuracy'].log(accuracy)
    #run['validation/epoch/dice_score'].log(dice_score)
    #run['validation/epoch/iou_score'].log(metrics_logs.get('iou_score'))
    #run['validation/epoch/precision_score'].log(metrics_logs.get('precision'))
    #run['validation/epoch/recall_score'].log(metrics_logs.get('recall'))

    model.train()


def save_predictions_as_imgs(loader,
                             model,
                             folder="saved_images/",
                             device='cpu'):
    """TODO: Docstring for save_predictions_as_imgs.

    :loader: TODO
    :model: TODO
    :folder: TODO
    :device: TODO
    :returns: TODO

    """
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.softmax(model(x))
            #preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
