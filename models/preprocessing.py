"""
This file contains all preprocessing code needed to manipulate the geographic
imagery for input into neural networks

Method taken from the rasterio docs: 
    https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
"""
from PIL import Image
import fiona
import rasterio
import time
import rasterio.mask
import cv2
import rioxarray
from rioxarray import merge
import numpy as np
from rasterio.plot import show
from imutils.perspective import four_point_transform
from image_slicer import slice
from image_slicer import save_tiles
import os
import shapely.vectorized
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shapely.geometry import shape

Image.MAX_IMAGE_PIXELS = 800000000

# =============================================================================
# combine tiff files to extract tiles
#
# The merging of these files subsequently removes all black values surrounding
# the field.
#
# This has allowed me to quickly create tiled training images without needing
# to delete all pixels with black values.
# =============================================================================

items = [
    '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R1C3_1.tif',
    '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R1C4_2.tif'
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R2C2_3.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R2C3_4.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R2C4_5.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R3C1_6.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R3C2_7.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R3C3_8.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R4C1_9.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R4C2_10.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R4C3_11.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R5C1_12.tif',
    # '/Users/jonthomann/Documents/research/projects/msc_thesis/drone_data/drone_images_labels/10meter_ortho_R5C2_13.tif'
]

elements = []

# iterate through items list and open each image
for val in items:
    elements.append(rioxarray.open_rasterio(val))

# merge all elements together
merged = merge.merge_arrays(elements, nodata=0.0)

# save merged image as tif file for later use
merged.rio.to_raster('./test_raster.tif')

# get pixel values of image
image = merged.values
