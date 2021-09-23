from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product
import rasterio as rio
from rasterio import windows

in_path = '../results/final_vis/pspnet/'
input_filename = 'pred_0.png'

out_path = '../results/final_vis/pspnet/'
output_filename = 'mask_tile_{}-{}.png'

def get_tiles(ds, width=512, height=1000):

    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off,
                                row_off=row_off,
                                width=width,
                                height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
    
    with rio.open(os.path.join(in_path, input_filename)) as inds:
        tile_width, tile_height = 256, 256

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(
                out_path,
                output_filename.format(int(window.col_off), int(window.row_off)))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))

def stitch_ground_truth():

    input = [
        '../results/fpn/ground_truth_images/Batch1/tile_13312-5120.tif',
        '../results/fpn/ground_truth_images/Batch1/tile_12800-512.tif',
        '../results/fpn/ground_truth_images/Batch1/tile_11776-512.tif',
        '../results/fpn/ground_truth_images/Batch1/tile_12800-3072.tif',
        '../results/fpn/ground_truth_images/Batch1/tile_11776-5120.tif',
    ]

    images = [Image.open(x) for x in input]
    w, h = zip(*(i.size for i in images))

    tw = sum(w)/2
    th = sum(h)/10

    final_im = Image.new('RGB', (int(tw), int(th)))

    offset = 0

    for im in images:
        im = im.resize((256,256))
        final_im.paste(im, (offset,0))
        offset += im.size[0]

    final_im.save('ground_truth_images_stitched.png')

def plot_all():
    input = [
        '../results/final_vis/groundtruth/tile_768-0.png',
        '../results/final_vis/groundtruth/tile_12800-3072.tif',
        '../results/final_vis/full-freq/mask_tile_768-0.png',
        '../results/final_vis/pspnet/mask_tile_1536-0.png',
        '../results/final_vis/fpn/mask_tile_1536-0.png',
        '../results/final_vis/PAN/mask_tile_1536-0.png',
        '../results/final_vis/LinkNet/mask_tile_768-0.png',

    ]
    images = [Image.open(x) for x in input]

    col = 7
    row = 1

    text = [
        'GT Image',
        'GT Mask',
        'MSWP',
        'PSP-Net',
        'FPN',
        'PAN',
        'LinkNet',
    ]

    fig = plt.figure(figsize=(5,18))
    
    col=7
    row=1

    plots = []
    i=0

    for image in images:
        plots.append(fig.add_subplot(row, col, i+1))
        plots[-1].set_title(text[i])
        plt.axis('off')
        plt.imshow(image)
        i += 1
    
    plt.show()
    




if __name__ == "__main__":
    plot_all()
