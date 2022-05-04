"""
This script creates the architecture for a wavelet-unet

The wavelet transform used takes a tensor as input and outputs the
LL, LH, HL, and HH sub-bands. LL is the appoximate image, LH extracts
horizontal features from images, HL gives vertical features, and the HH 
subband outputs the diagonal features from an input image. 
"""

import torch
from torch._C import dtype
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
import torchvision.models as models
import time
from torch.autograd import Variable


class DwtPyramidBlock(nn.Module):
    def __init__(self, J, mode, wave, depth, og_size_list):
        super(DwtPyramidBlock, self).__init__()
        self.J = J
        self.mode = mode
        self.wave = wave
        self.depth = depth
        self.og_size_list = og_size_list

        self.dwtList = []

        self.dwtf = DWTForward(J=self.J, wave=self.wave, mode=self.mode).cuda()
    
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
                )  # .contiguous() - this might be needed depending on if pytorch yells at me :(
            
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
            self.dwtList = []  # IMPORTANT, reset list every time a pyramid is built otherwise the model will run out of memory
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



class DtcwtPyramidBlock(nn.Module):
    """
    This method is similar to the above wavelet pyramid method, however it uses 
    the dual-tree complex wavelet, which is a shift invariant wavelet transfor that
    outputs six subbands rather than three. 

    What makes this wavelet interesting to use, is that not only is it able to hold real
    numbers, but imaginary as well. 

    The sub-bands represent 15, 45, 75, 105, 135, and 165 degree wavelets.
    """
    def __init__(self, biort : str, q_shift : str):
        super(DtcwtPyramidBlock, self).__init__()
        self.biort = biort
        self.q_shift = q_shift
    
    def forward(self, x : Tensor):
        print(x)


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

        # initialize wavelet transforms for use in forward
        self.dwtF = DWTForward(J=3, mode='zero', wave='haar').cuda()

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

        # load backbone model se-resnext to extract meaningful features from the start
        self.pretrained(x)

        # for each skip connection, there is also one wavelet connection to
        # decrease information loss
        skip_connections = []
        # wavelet_skips = []
        # x1_hh = x  # double skip connection
        # temp_var = 0

        for down in self.downs:
            # separate x into two variables to perform pooling and DWT on the features

            # down-scale using pooling operations, these will be the features passed down the nextwork as well as across
            x = down(x)
            # x1_hh = down(x1_hh)  # get same channel size

            # downscale using DWT, these features are not propogated down the network
            # they are used in tandem with skip connections to create more meaningful
            # data representations when we are upscaling images back to their original

            #print('---------------x: ', x.shape)
            #print('---------------x1: ', x1_hh.shape)
            
            skip_connections.append(x)

            # to increase spatial information gain wavelet decompositions will be used
            # alongside pooling operations and concatenated with everything at the end
            x = self.pool(x)
        
        x = self.bottleneck(x)

        # call wavelet pyramid at bottom of u-net architecture
        pyramid = self.dwt_pyramid_block(x)
        x = torch.cat((x, pyramid), dim=1)  # concat pyramid and original input x

        # now due to the large amount of channels, x should be put through a projection layer
        # to reduce the number of feature channels in the architecture.
        x = self.projection_layer(x)


        skip_connections = skip_connections[::-1]
        # wavelet_skips = wavelet_skips[::-1]
        # temp2 = 0

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        x = self.final_conv(x)
        return x


def testArch():
    """testing the unet architecture
    :returns: TODO
    """
    x = torch.randn((3, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=6)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    testArch()
