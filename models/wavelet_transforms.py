"""
File that calculates different wavelet transforms on pytorch tensor objects.

Current wavelet transforms:
    Haar
    Dual-Tree Complex WT

PytorchWavelets is the library used to do the calculations:
    https://github.com/fbcotter/pytorch_wavelets
    https://doi.org/10.17863/CAM.53748
"""
import torch
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse


def dwt_forward(tensor, wave, mode, J):
    """This method takes a pytorch tensor and manipulates it using a discrete
    wavelet transform. 

    :tensor: a pytorch tensor of any size
    :wave: the wavelet transform to use
    :mode: signal extrapolation mode
    :returns: Yl, Yh[]

    """
    dwt = DWTForward(J=J, wave=wave, mode=mode)
    Yl, Yh = dwt(tensor)

    print(Yl.shape)
    print(Yh[0].shape)
    print(Yh[1].shape)
    print(Yh[2].shape)
    return Yl, Yh


def dwt_inverse(Yl, Yh, wave, mode):
    """method that calculates the inverse of the wavelet transform

    :Yl: lowpass filters
    :Yh: array of highpass coefficients
    :wave: wavelet transform to use
    :mode: signal extrapolation mode
    :returns: returns original tensor

    """
    iwt = DWTInverse(wave=wave, mode=mode)
    Y = iwt((Yl, Yh))
    return Y


def dtcwt_forward(tensor, biorthogonal, qshift, J):
    """This method takes a tensor and calculates the dual-tree complex wavelet
    transform. The method was found in a paper written by Ivan W. Selesnick, 
    Richard G. Baraniuk, and Nick G. Kingsbury in 2005 and could have some
    advantages over a DWT for extremely high resolution geometric imagery.

    :tensor: a tensor of size (N, C, H, W)
    :biorthogonal: first level biorthogonal wavelet filters
    :qshift: second level quarter shift filters
    :J: number of levels of decomposition
    :returns: (yl, yh)
    tuple of lowpass, yl, and bandpass, yh, coefficients

    """
    dtcwt = DTCWTForward(J=J, biort=biorthogonal, qshift=qshift)
    Yl, Yh = dtcwt(tensor)

    print(Yl.shape)
    print(Yh[0].shape)
    print(Yh[1].shape)
    print(Yh[2].shape)
    return Yl, Yh


def dtcwt_inverse(yl, yh, biorthogonal, qshift):
    """method calculates the inverse of the dual-tree complex wavelet transform


    :yl: lowpass coefficients
    :yh: bandpass coefficients
    :biorthogonal: first level biorthogonal wavelet filters
    :qshift: second level quarter shift filters
    :returns: original tensor

    """
    inv_dtcwt = DTCWTInverse(biort=biorthogonal, qshift=qshift)
    Y = inv_dtcwt((yl, yh))
    return Y


if __name__ == '__main__':
    Yl, Yh = dwt_forward(torch.randn(10, 5, 64, 64), 'haar', 'zero', 3)
    Y = dwt_inverse(Yl, Yh, 'haar', 'zero')
    print(Y.shape)
