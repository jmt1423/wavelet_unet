from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

im = Image.open('../../newtile_0.tif')
im_np = np.array(im)

og_transform = A.Compose([
    A.Resize(height=256, width=256),
])

transform1 = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=35, p=1.0),
])

transform2 = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=1.0),
])

transform3 = A.Compose([
    A.Resize(height=256, width=256),
    A.VerticalFlip(p=1.0),
])

transform4 = A.Compose([
    A.Resize(height=256, width=256),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=1.0),
])

transform5 = A.Compose([
    A.Resize(height=256, width=256),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=1.0),
])

transform6 = A.Compose([
    A.Resize(height=256, width=256),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
])

transform7 = A.Compose([
    A.Resize(height=256, width=256),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
])

og_aug = og_transform(image=im_np)
aug1 = transform1(image=im_np)
aug2 = transform2(image=im_np)
aug3 = transform3(image=im_np)
aug4 = transform4(image=im_np)
aug5 = transform5(image=im_np)
aug6 = transform6(image=im_np)
aug7 = transform7(image=im_np)

og_final = Image.fromarray(og_aug['image'])
final1 = Image.fromarray(aug1['image'])
final2 = Image.fromarray(aug2['image'])
final3 = Image.fromarray(aug3['image'])
final4 = Image.fromarray(aug4['image'])
final5 = Image.fromarray(aug5['image'])
final6 = Image.fromarray(aug6['image'])
final7 = Image.fromarray(aug7['image'])

all_images = [og_final, final1, final2, final3, final4, final5, final6, final7]
text = [
    'Original', 
    'Rotate', 
    'Horizontal Flip', 
    'Vertical Flip', 
    'Shift Scale Rotate', 
    'Color Jitter', 
    'RGB Shift', 
    'Brightness Contrast'
]

fig = plt.figure(figsize=(5,5))
col=4
row=2

plots = []
i=0

for image in all_images:
    plots.append(fig.add_subplot(row, col, i+1))
    plots[-1].set_title(text[i])
    plt.imshow(image)
    i += 1

plt.show()


