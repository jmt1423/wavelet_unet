"""
This script creates the architecture for a wavelet-unet

The wavelet transform used takes a tensor as input and outputs the
LL, LH, HL, and HH sub-bands. LL is the appoximate image, LH extracts
horizontal features from images, HL gives vertical features, and the HH 
subband outputs the diagonal features from an input image. 
"""

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
import torchvision.models as models
import time



class DoubleConv(nn.Module):
    """Docstring for DoubleConv. """
    def __init__(self, in_channels, out_channels):
        """TODO: to be defined. """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """feed forward network for u-net
        :x: TODO
        :returns: TODO
        """
        return self.conv(x)


class UNETablation(nn.Module):
    """Docstring for UNET. """
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        features=[64, 128, 256, 512],
    ):
        """TODO: to be defined.
        :in_channels: TODO
        :out_channels: TODO
        :features: TODO
        :128: TODO
        :256: TODO
        :512]: TODO
        """
        super(UNETablation, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.pretrained = models.resnext50_32x4d(pretrained=True)

        # Down scaling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up scaling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                ))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """feed forward method
        :x: TODO
        :returns: TODO
        """

        # load backbone model se-resnext to extract meaningful features from the start
        self.pretrained(x)

        skip_connections = []


        for down in self.downs:
            x = down(x)
            
            skip_connections.append(x)

            x = self.pool(x)


        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def testArch():
    """testing the unet architecture
    :returns: TODO
    """
    x = torch.randn((3, 256, 256, 3))
    model = UNETablation(in_channels=1, out_channels=3)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    testArch()