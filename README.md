# Wavelet-Unet and Image segmentation #

* Python version: 3.8.8
* Environment was created using pyenv and pyenv-virtualenv together

to create requirements.txt using ALL pip3 installed packages:
This may not work completely as the environment used came from lambdastack

```
pip3 freeze > requirements.txt
```

Training hardware:
Intel® Core™ i7-11700F Processor
NVIDIA GeForce RTX 3070 8 GB
RAM: 16 GB / Storage: 1 TB HDD & 512 GB SSD

All packages used in this project (except for pytorch_wavelets) can be downloaded using:
[LambdaStack](https://lambdalabs.com/lambda-stack-deep-learning-software)

Main packages used:
[Pytorch](https://pytorch.org/)
[pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)
[pytorch_segmentation](https://github.com/qubvel/segmentation_models.pytorch)
[rasterio](https://rasterio.readthedocs.io/en/latest/)
[rioxarray](https://corteva.github.io/rioxarray/stable/)

All metrics were logged using:
[Neptune ai](https://neptune.ai/)

NOTE: 
the package pytorch_wavelets was modified due to casting and backpropagation errors, in lowlevel.py several of the variables had to be casted to float values manually due to an issue in the training process. I am not exactly what causes this when the pyramid block is inserted into the U-net architecture. 

The second issue was within the recursive function, as the tensor object is passed several times through the same function it might cause issues with modifying values in-place, to correct this issue the input tensor to the pyramid must be detached from pytorch memory before calculating any discrete wavelet transforms.

