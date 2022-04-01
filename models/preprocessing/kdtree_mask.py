"""
This script takes a masked image from labelbox's API (where I labelled the original image)
and uses a KDTree algorithm from scipy to threshold each pixel into a given colorspace.
"""
from PIL import Image
import torch
import time
import config

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



# Project ID of blackpool labels
PROJECT_ID = config.PROJECT_ID
# endpoint of labelbox api
ENDPOINT = "https://api.labelbox.com/graphql"

# client key this needs to be removed and 
client = Client(api_key=config.LB_API_KEY, endpoint=ENDPOINT)
project = client.get_project('ckybon3yzbrb30zai53rbfhic')

labels = project.label_generator()

# Create a mapping for the colors
hex_to_rgb = lambda hex_color: tuple(int(hex_color[i+1:i+3], 16) for i in (0, 2, 4))
colors1 = {tool.name: hex_to_rgb(tool.color) for tool in OntologyBuilder.from_project(project).tools}

# Grab the first label and corresponding image
label = next(labels)
image_np = label.data.value

# Draw the annotations onto the source image
for annotation in label.annotations:
    if isinstance(annotation.value, Geometry):
        image_np = annotation.value.draw(canvas = image_np, color = colors1[annotation.name], thickness = 5)
final = Image.fromarray(image_np.astype(np.uint8))


# using kdtree to reduce color space of mask image to create correct color mapping for model
REDUCED_COLOR_SPACE = True

# borrow a list of named colors from matplotlib
if REDUCED_COLOR_SPACE:
    use_colors = {k: colors.cnames[k] for k in ['red', 'blue', 'yellow', 'pink', 'purple', 'green']}
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