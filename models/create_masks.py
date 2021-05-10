import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import fiona
import rasterio
import rasterio.mask
import time

Image.MAX_IMAGE_PIXELS = 400000000

# creating masks for segmentation
with fiona.open("../drone_data/flower_shapefile_1_2/YellowFlowers1.shp",
                "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open("../drone_data/10meter_ortho_R1C4.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)

im = Image.open("RGB.byte.masked.tif")
im.show()
# im = Image.open('../drone_data/10meter_ortho_R1C3.tif')
# im.show()

# im2 = Image.open('../drone_data/10meter_ortho_R1C4.tif')
# im2.show()

# these images show nothing but black image.
# im3 = Image.open('../drone_data/10meter_ortho_R1C3_GT.tif')
# im3.show()

# im4 = Image.open('../drone_data/10meter_ortho_R1C4_GT.tif')
# im4.show()
