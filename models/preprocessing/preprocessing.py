"""
This file contains all preprocessing code needed to manipulate the geographic
imagery for input into neural networks

Method taken from the rasterio docs: 
    https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
"""
from PIL import Image
import fiona
import rasterio as rio
import time
import rasterio.mask
import cv2
import rioxarray
from rioxarray import merge
import numpy as np
from rasterio.plot import show
import os
from rasterio import windows
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
def merge_files():
    """
    Method to merge all tiff files into one for splitting
    :returns: TODO

    """

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
    # image = merged.values


# =============================================================================
# Mask all tiled images
# =============================================================================
with fiona.open(
        '../../../Yellow_flower_label/YellowFlowers4.shp'
) as shapefile:
    shapes1 = [feature['geometry'] for feature in shapefile]

# with fiona.open(
#        '../../drone_data/drone_images_labels/Yellow_flower_label/YellowFlowers2.shp'
# ) as shapefile:
#    shapes2 = [feature['geometry'] for feature in shapefile]


def create_mask(input_file, output_file, shapes1):
    """
    method to create a mask of a given input image

    :input_file: image to mask
    :output_file: name of file to save
    :returns: masked image

    """
    while True:
        try:
            with rasterio.open('../../../images/' +
                               input_file) as src:
                out_image, out_transform = rasterio.mask.mask(src,
                                                              shapes1,
                                                              crop=True)
                out_meta = src.meta
            out_meta.update({
                'driver': 'GTiff',
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            with rasterio.open(input_file, 'w', **out_meta) as dest:
                dest.write(out_image)
            break
        except ValueError:
            print('shapefile does not match raster')
            # shapes1, shapes2 = shapes2, shapes1


num = 0
for item in os.listdir('../../../images/'):
    splitname = os.path.splitext(item)[0]
    combined = splitname + '_' + str(num) + '.tif'
    print(item)
    create_mask(item, combined, shapes1)

    num += 1
