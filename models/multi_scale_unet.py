"""
This model creates the architecture for a wavelet-unet

The wavelet transform used takes a tensor as input and outputs the
LL, LH, HL, and HH sub-bands. LL is the appoximate image, LH extracts
horizontal features from images, HL gives vertical features, and the HH 
subband outputs the diagonal features from an input image. 
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
from pytorch_wavelets import DWTForward
import torchvision.models as models


class DwtPyramidBlock(nn.Module):
    def __init__(self, J, mode, wave, depth, og_size_list):
        super(DwtPyramidBlock, self).__init__()
        self.J = J
        self.mode = mode
        self.wave = wave
        self.depth = depth
        self.og_size_list = og_size_list

        self.dwtList = []
        if torch.cuda.is_available():
            self.dwtf = DWTForward(J=self.J, wave=self.wave, mode=self.mode).cuda()
        else:
            self.dwtf = DWTForward(J=self.J, wave=self.wave, mode=self.mode)

    def forward(self, x):
        """
        This recursive forward pass function create a multi-scale representation 
        of a given tensor object x using wavelet transforms.
        """
        if self.depth < 2:  # recursive step
            yl, yh = self.dwtf(x.detach())  # get wavelet coefficients
            feat = torch.unbind(yh[0], dim=2)  # unbind scale 1 band
            feat0 = feat[0] # get features (yl, yh -> (lh, hl, hh))
            #feat1 = feat[1]
            #feat2 = feat[2]
            self.dwtList.append(yl)  # append to list for later use
            #self.dwtList.append(feat1)
            #self.dwtList.append(feat2)
            #self.dwtList.append(yl)
            self.depth += 1
            return self.forward(feat0)  # create next layer of pyramid
        elif self.depth == 2:  # end of pyramid
            yl, yh = self.dwtf(x.detach())  # get final coefficients
            feat = torch.unbind(yh[0], dim=2)
            feat0 = feat[0]
            #feat1 = feat[1]
            #feat2 = feat[2]
            self.dwtList.append(yl)  # append to list for later use
            #self.dwtList.append(feat1)
            #self.dwtList.append(feat2)
            #self.dwtList.append(yl)

            for i in range(len(self.dwtList)):  # pad each element in list for concatenation later
                p_size = int(self.get_pad_size(self.dwtList[i]))  # gets the variable needed for padding
                p_size_1 = p_size
                temp0 = list(self.dwtList[i].shape)[3]
                if(temp0 == 1):
                    p_size_1 += 1

                self.dwtList[i] = F.pad(  # pad list item i
                    self.dwtList[i],
                    (p_size, p_size_1, p_size, p_size_1),
                    'constant',  # zero padding so constant value of 0's
                    0
                )  # .contiguous()
            
            # after padding is done, now we can concat all tensor objects together
            final_pyramid = torch.cat((
                self.dwtList[0],
                self.dwtList[1],
                self.dwtList[2],
                #self.dwtList[3],
                #self.dwtList[4],
                #self.dwtList[5],
                #self.dwtList[6],
                #self.dwtList[7],
                #self.dwtList[8],
                #self.dwtList[9],
                #self.dwtList[10],
                #self.dwtList[11],
            ), dim=1)
            self.dwtList = []
            self.depth = 0  # same as above.
            return final_pyramid

    def get_pad_size(self, x):
        """
        method that takes the original tensor object height and
        the given wavelet decomposed tensor to calculate the size 
        needed to pad each tensor object before a torch.cat() is 
        used.

        Formula:
        Given two tensors of size [B_1, C_1, H_1, W_1] and
        [B_2, C_2, H_2, H_3] -->

        to find the needed values to pad take:

        (H1 - H2) / 2 = p_size -> {H1 > H2}

        """
        h1 = self.og_size_list[3]
        h2 = list(x.shape)[3]

        return (h1-h2)/2


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

class LinearConv(nn.Module):
    """Docstring for LInearConv. """
    def __init__(self, in_channels, out_channels):
        """TODO: to be defined. """
        super(LinearConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """feed forward network for u-net
        :x: TODO
        :returns: TODO
        """
        return self.conv(x)


class UNET(nn.Module):
    """Docstring for UNET. """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
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
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if torch.cuda.is_available():
            # initialize wavelet transforms for use in forward
            self.dwtF = DWTForward(J=3, mode='zero', wave='haar').cuda()
        else:
            self.dwtF = DWTForward(J=3, mode='zero', wave='haar')

        self.dwt_pyramid_block = DwtPyramidBlock(J=3, mode='zero', wave='haar', depth=0, og_size_list=[0, 0, 16, 16])

        self.projection_layer = LinearConv(4096, 1024)

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
        self.pretrained(x)

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        # call wavelet pyramid at bottom of u-net architecture
        pyramid = self.dwt_pyramid_block(x)
        x = torch.cat((x, pyramid), dim=1)  # concat pyramid and original input x

        x = self.projection_layer(x)


        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        x = self.final_conv(x)
        return x