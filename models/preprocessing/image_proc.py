import cv2
import numpy as np
from PIL import Image
import scipy.spatial as sp


import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

main_colors = [(0, 0, 255),
               (225, 0, 225),
               (255, 0, 0),
               (255, 225, 225),
               (255, 255, 0),]
# sample image
#np.random.seed(1)
#avg = np.random.randint(0,255, (10,10,3), dtype=np.uint8)

# in your code, make sure `avg` is an np array
# you can use `cv2.imread` for that purpose, not the BGR color space
# for example
avg = cv2.imread('./bp_validation_large_again.jpg')
avg = cv2.cvtColor(avg, cv2.BGR2RGB) # convert to RGB

# convert main_colors to np array for indexing
main_colors = np.array(main_colors)

# compute the distance matrix
dist_mat = sp.distance_matrix(avg.reshape(-1,3), main_colors)

# extract the nearest color by index
color_idx = dist_mat.argmax(axis=1)

# build the nearest color image with indexing
nearest_colors = main_colors[color_idx].reshape(avg.shape)

# plot
fig, axes = plt.subplots(1,2)
axes[0].imshow(avg)             # original image
axes[1].imshow(nearest_colors)  # nearest_color
plt.show()