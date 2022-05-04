import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from PIL import Image


mask = Image.open('../../bp_validation_large.jpg').convert('RGB')

h=1467
w=316
#img.show()

target = torch.from_numpy(np.array(mask))
colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
target = target.permute(2, 0, 1).contiguous()

mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

mask = torch.empty(h, w, dtype=torch.long)
print(colors)

for k in mapping:
    # Get all indices for current class
    idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
    validx = (idx.sum(0) == 3)  # Check that all channels match
    mask[validx] = torch.tensor(mapping[k], dtype=torch.long)

print(mask)