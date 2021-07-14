"""
This file creates a multi-resolution wavelet pyramid that can integrate with
most modern image segmentation architectures.

Given a tensor of size (N, C, H, W) a wavelet transform (in this current case,
I am using a Haar basis) is then performed to create two new tensor objects.

Yl is a tensor that containts the LL band and Yh is a tensor list containing
the LH (horizontal), HL (vertical), and HH (diagonal) coefficients for each
scale. 

Yl should be of size (N, C, H, W)
Yh should be of size (N, C, 3, H, W)
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse


class ConvManipulation(nn.Module):
    """
    Class to manipulate the resulting feature maps of the wavelet pyramid
    """
    def __init__(self, in_channels, out_channels):
        """
        """
        super(ConvManipulation, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :x: data passing through the convolution layer
        :returns: manipulated data

        """
        x = self.conv2d(x)
        x = self.relu(x)
        return x


def create_lowband_haar_pyramid(input_tensor, wave, mode, J, og_size):
    """this method creates a multi-resolution wavelet pyramid. The
    goal of this neural network block is to capture multi-scale information of
    high resolution images with as little information loss as possible. 

    Combined with using wavelet transforms as downscaling and upscaling
    opertations which allow a neural network to retain high quality feature 
    maps, a cascading wavelet pyramid can also be added into an architecture. 

    :tensor: feature map tensor for creation of the pyramid
    :wave: wavelet basis
    :mode: signal extrapolation mode 
    :J: number of levels of decomposition
    :returns: a fused multi-resolution feature map

    """
    dwt = DWTForward(J=J, wave=wave, mode=mode)
    conv_manipulation = ConvManipulation(1, 1)
    upsample = nn.UpsamplingBilinear2d(size=(og_size[0], og_size[1]))

    Yl_0 = input_tensor
    Yl_1, Yh_1 = dwt(Yl_0)
    Yl_2, Yh_2 = dwt(Yl_1)
    Yl_3, Yh_3 = dwt(Yl_2)

    Yl_0_conv = conv_manipulation(Yl_0)
    Yl_1_conv = conv_manipulation(Yl_1)
    Yl_2_conv = conv_manipulation(Yl_2)
    Yl_3_conv = conv_manipulation(Yl_3)

    Yl_0_up = upsample(Yl_0_conv)
    Yl_1_up = upsample(Yl_1_conv)
    Yl_2_up = upsample(Yl_2_conv)
    Yl_3_up = upsample(Yl_3_conv)

    final_feature_map = torch.cat((Yl_0_up, Yl_1_up, Yl_2_up, Yl_3_up), dim=1)
    return final_feature_map


if __name__ == '__main__':
    tensor = torch.randn(3, 1, 400, 400)
    wave = 'haar'
    mode = 'zero'
    J = 3

    map = create_lowband_haar_pyramid(tensor, wave, mode, J, (500, 500))
    print(map.shape)
