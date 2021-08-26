# Image Segmentation of Flower Resources #

* Python version I started with: 3.8.8
* My environment was created using pyenv and pyenv-virtualenv together

to create requirements.txt using ALL pip3 installed packages:

```
pip3 freeze > requirements.txt
```

NOTE: 
the package pytorch_wavelets was modified due to casting and backpropagation errors, in lowlevel.py several of the variables had to be casted to float values manually due to an issue in the training process. I am not exactly what causes this when the pyramid block is inserted into the U-net architecture. 

The second issue was within the recursive function, as the tensor object is passed several times through the same function it might cause issues with modifying values in-place, to correct this issue the input tensor to the pyramid must be detached from pytorch memory before calculating any discrete
wavelet transforms.

