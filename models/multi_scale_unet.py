import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse


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
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
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

        # initialize wavelet parameters
        self.wave = 'haar'
        self.mode = 'zero'
        self.J = 3
        self.dwtf = DWTForward(self.J, self.wave, self.mode)
        self.idwt = DWTInverse(self.mode, self.wave)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # TODO: Remove maxpool and replace with wavelet transform methods
        # that have been implemented in wavelet_transforms.py
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down scaling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up scaling
        # TODO: replace with inverse discreet wavelet transform
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
        # lists of tensor objects containing lowpass and highpass
        # filter coefficients
        wave_feature_connector = []

        # before using DWT and IDWT put the intitial tensor through a double
        # conv block to extract features.
        x = self.downs[0](x)

        print("=============================================")
        # Calculate DWT
        Yl, Yh = self.dwtf(x)
        print(Yl.shape)
        print(Yh[0].shape)
        print(Yh[1].shape)
        print(Yh[2].shape)

        # add new dimension to Yl coefficients for stacking
        Yl = Yl[:, :, None, :, :]

        stacked_tensor = torch.stack([Yl, Yh])
        print(stacked_tensor.shape)
        print("stack success!")

        print("=============================================")

        # split Yh and concatenate coefficients

        # looping one less time as we have already gone through a doubleconv
        # and DWT layer
        for down in self.downs[1:]:
            # use concatenated features as input into doubleconv
            Yh = down(Yh)
            Yl = down(Yl)

            # add features into connector
            skip_connection_lowpass.append(Yl)
            skip_connection_highpass.append(Yh)

            # x = self.pool(x)
            # calculate dwt transform of concatenated features

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
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    testArch()
