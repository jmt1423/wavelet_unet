import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

import torch

# Create dummy target image
mask = Image.open('../../../testing.png').convert('RGB')
target2 = Image.open('./final_image.jpg')
target2.show()

#print(target)

pixels = mask.load()
for i in range(mask.size[0]):
    for j in range(mask.size[1]):
            if pixels[i,j] == (255, 225, 225):
                pixels[i,j] = pixels[i,j]
            elif pixels[i,j] == (255, 225, 226):
                pixels [i,j] = (255,225,225)
            elif pixels[i,j] == (255, 225, 241):
                pixels [i,j] = (255,225,225)
            else: pixels[i,j] = pixels[i,j]
mask.save('example.png')

time.sleep(1000)