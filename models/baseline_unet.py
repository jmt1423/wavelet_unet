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

class DwtPyramidBlock(nn.Module):
    """
    This class takes as input a tensor of arbirtrary size, as the architecture
    should be able to be plugged into other models, and runs it through a series of 
    wavelet transforms that essentially down sample the image into four different bands
    and three different scales.

    With computation time considered, only the highpass scale 1 bandpass coefficients are
    passed down the pyramid to increase multi-scale information gain.

    Wavelet families tested: Haar, Daubechies, Complex Morlet Wavelets, Symlets
    """

    def __init__(self, family : str, mode : str):
        super(DwtPyramidBlock, self).__init__()
        self.family = family
        self.mode = mode

    def forward(self, x : Tensor):
        print('hi')

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


class ProjectionLayer(nn.Module):
    """
    This class is just contains a single 1x1 convolution layer. Otherwise known
    as a projection layer, this allows me to adjust the depth of all feature
    maps at will.
    The main usage of this class is to match the feature map depth of the 
    encoder to the decoder when passing the transformed data through the 
    skip connections.
    """
    def __init__(self, in_channels, out_channels):
        """
        initialize projection layer
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.softconv = nn.Conv2d(self.in_channels,
                                  self.out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass data into the convolution layer, followed by a ReLU
        non-linearity
        :x: tensor object being passed into convolution layer
        :returns: projected feature map
        """
        x = self.softconv(x)
        x = self.relu(x)
        return x


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
        super(UNETablation, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # initialize wavelet transforms for use in forward
        self.dwtF = DWTForward(J=3, mode='zero', wave='haar').cuda()
        self.dtcwtF = DTCWTForward(
            biort='near_sym_a', 
            qshift='qshift_a',
            J=3,
            mode='symmetric'
        ).cuda()

        self.pretrained = models.resnext50_32x4d(pretrained=True)
        self.projection_layer = ProjectionLayer(5120, 1024)

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
            
            skip_connections.append(x)

            # to increase spatial information gain wavelet decompositions will be used
            # alongside pooling operations and concatenated with everything at the end
            x = self.pool(x)

            # if temp_var > 10000:
            #     yl, yh = self.dwtF(x1_hh)

            #     yh = torch.unbind(yh[0], dim=2)
            #     x1_hh = yh[0]
            #     wavelet_skips.append(x1_hh)

            # temp_var += 1


        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        # wavelet_skips = wavelet_skips[::-1]
        # temp2 = 0

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # if temp2 < 3:  # wavelet skip connection, not done on first iteration as there is one less connection
            #     wavelet_skip = wavelet_skips[temp2]

            #     if x.shape != wavelet_skip.shape:
            #         x = TF.resize(x, size=wavelet_skip.shape[2:])
                
            #     concat_skip = torch.cat((wavelet_skip, x), dim=1)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            # temp2 += 1

        return self.final_conv(x)


def testArch():
    """testing the unet architecture
    :returns: TODO
    """
    x = torch.randn((3, 1, 160, 160))
    model = UNETablation(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    testArch()