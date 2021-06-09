"""

This script creates masks of the two separate 
.tif images using rasterio and fiona.

To create the raster masks yourself the files must match:
    YellowFlowers1.shp == 10meter_ortho_R1C3
    YellowFlowers2.shq == 10meter_ortho_R1C4

Method taken from the rasterio docs: 
    https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
"""
from PIL import Image
import fiona
import rasterio
import time
import rasterio.mask
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from image_slicer import slice
from image_slicer import save_tiles
import os
import shapely.vectorized
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shapely.geometry import shape

Image.MAX_IMAGE_PIXELS = 400000000

# test_image = Image.open('../drone_data/train_images/_01_01.tiff')
# test_image.show()
# time.sleep(100)
# split image for training into 100 sections
# tiles = slice('../drone_data/10meter_ortho_R1C3.tif', 100, save=False)
# save_tiles(tiles, '../drone_data/train_images/', format='TIFF')
# print(type(tiles[0]))
# print('done')
# time.sleep(100)

# img = cv2.imread('../drone_data/train_images/10meter_ortho_R1C3.tif')

# print('done')
# train_image = Image.open('../drone_data/val_images/10meter_ortho_R1C4.tif')
# print(train_image.size)
# train_image.show()
# train_image.show()
# train_mask = Image.open('../drone_data/val_masks/10meter_ortho_R1C4.tif')
# print(train_mask.size)
# train_mask.show()

# creating masks for segmentation
with fiona.open("../drone_data/drone_images_labels/background/Background1.shp",
                "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open(
        "../drone_data/drone_images_labels/10meter_ortho_R1C3_1.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

plt.imshow(out_image)

# with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
#     dest.write(out_image)

# show the newly created masked raster image
# im = Image.open("RGB.byte.masked.tif")
# im.show()
