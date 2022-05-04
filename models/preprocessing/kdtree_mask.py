"""
This script takes a masked image from labelbox's API (where I labelled the original image)
and uses a KDTree algorithm from scipy to threshold each pixel into a given colorspace.
"""
from PIL import Image
import torch
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import io
from skimage import color
from skimage import segmentation

from labelbox import Client, OntologyBuilder
from labelbox.data.annotation_types import Geometry
from getpass import getpass
from PIL import Image
import numpy as np
import os

from matplotlib import colors
from scipy.spatial import cKDTree as KDTree
from scipy.misc import face


final = Image.open('./bp_validation_large_again.jpg')


# using kdtree to reduce color space of mask image to create correct color mapping for model
REDUCED_COLOR_SPACE = True

# borrow a list of named colors from matplotlib
if REDUCED_COLOR_SPACE:
    use_colors = {k: colors.cnames[k] for k in ['red', 'blue', 'yellow', 'white', 'purple', 'green']}
else:
    use_colors = colors.cnames

# translate hexstring to RGB tuple
named_colors = {k: tuple(map(int, (v[1:3], v[3:5], v[5:7]), 3*(16,)))
                for k, v in use_colors.items()}
ncol = len(named_colors)

if REDUCED_COLOR_SPACE:
    ncol -= 1
    no_match = named_colors.pop('green')
else:
    no_match = named_colors['green']

# make an array containing the RGB values 
color_tuples = list(named_colors.values())
color_tuples.append(no_match)
color_tuples = np.array(color_tuples)

color_names = list(named_colors)
color_names.append('no match')


# get picture
img = final

# build tree
tree = KDTree(color_tuples[:-1])

# tolerance for color match `inf` means use best match no matter how bad it is
tolerance = np.inf

# find closest color in tree for each pixel in picture
dist, idx = tree.query(img, distance_upper_bound=tolerance)

# count and reattach names
counts = dict(zip(color_names, np.bincount(idx.ravel(), None, ncol+1)))

print(counts)

PIL_image = Image.fromarray(np.uint8(color_tuples[idx])).convert('RGB')
PIL_image.show()