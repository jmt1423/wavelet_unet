"""
This file implements all custom loss functions used in the testing
of the novel model.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

"""
calculates the similarity between two images using:

DL(y, p) = 1 - ((2yp+1)/(y+p+1))
"""
class DiceLoss(nn.Module):
    def __init__(self, weight=None, average_size=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1-dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_coeff = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        DiceBCE = BCE + dice_coeff
        return DiceBCE

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        """
       I am assuming the model does not have sigmoid layer in the end. if that is the case, change torch.sigmoid(logits) to simply logits
        """
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score