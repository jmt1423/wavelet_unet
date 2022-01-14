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

#img = Image.open('../../../CoastSat/data/blackpool/images/bp_labels.jpg').convert('1')
#img.save('../../../CoastSat/data/blackpool/images/label_greyscale.png')
# Pick a project that has and of box, point, polygon, or segmentation tools tools in the ontology
# and has completed labels
PROJECT_ID = "ckybon3yzbrb30zai53rbfhic"
# Only update this if you have an on-prem deployment
ENDPOINT = "https://api.labelbox.com/graphql"

client = Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3lib2o1dmtibTIwMHphOTY1ajhjdjE2Iiwib3JnYW5pemF0aW9uSWQiOiJja3lib2o1dmNibTF6MHphOWZkbzg1M2MzIiwiYXBpS2V5SWQiOiJja3lkM2cxNnc4czhkMHo3dDFjdHNmZ2JzIiwic2VjcmV0IjoiY2RiOTU3OWFjYjIzMTQ4Yjc4MGU4ZWQzOWIxODE2M2QiLCJpYXQiOjE2NDIwODU3MjgsImV4cCI6MjI3MzIzNzcyOH0.tZQkFFY2wa3v_ia6dnDGzDPnyDAIJtrGkGr12h6AFJQ', endpoint=ENDPOINT)
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
#final.show()


# using kdtree to reduce color space of mask image to create correct color mapping for model

REDUCED_COLOR_SPACE = True

# borrow a list of named colors from matplotlib
if REDUCED_COLOR_SPACE:
    use_colors = {k: colors.cnames[k] for k in ['red', 'green', 'blue', 'yellow', 'pink', 'purple']}
else:
    use_colors = colors.cnames

# translate hexstring to RGB tuple
named_colors = {k: tuple(map(int, (v[1:3], v[3:5], v[5:7]), 3*(16,)))
                for k, v in use_colors.items()}
ncol = len(named_colors)

if REDUCED_COLOR_SPACE:
    ncol -= 1
    no_match = named_colors.pop('purple')
else:
    no_match = named_colors['purple']

# make an array containing the RGB values 
color_tuples = list(named_colors.values())
color_tuples.append(no_match)
color_tuples = np.array(color_tuples)

color_names = list(named_colors)
color_names.append('no match')

# get example picture
img = final

# build tree
tree = KDTree(color_tuples[:-1])
# tolerance for color match `inf` means use best match no matter how
# bad it may be
tolerance = np.inf
# find closest color in tree for each pixel in picture
dist, idx = tree.query(img, distance_upper_bound=tolerance)
# count and reattach names
counts = dict(zip(color_names, np.bincount(idx.ravel(), None, ncol+1)))

print(counts)

import pylab

pylab.imshow(img)
pylab.savefig('orig.png')
pylab.clf()
pylab.imshow(color_tuples[idx])
pylab.savefig('minimal.png' if REDUCED_COLOR_SPACE else 'reduced.png')
