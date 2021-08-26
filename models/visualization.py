from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        '../results/ground_truth_images_stitched.png',
        '../results/Wavelet-Unet/full-frequency/ground_truth_masks/0.png',
        '../results/Wavelet-Unet/full-frequency/predictions/pred_0.png',
        '../results/PSPNet/predictions/pred_0.png',
        '../results/fpn/predictions/pred_0.png',
        '../results/PAN/predictions/pred_0.png',
        '../results/LinkNet/predictions/pred_0.png'
    ]
    images = [Image.open(x) for x in input]

    col = 1
    row = 7

    text = [
        'Ground Truth Images',
        'Ground Truth Masks',
        'Full Frequency Wavelet Pyramid',
        'PSP-Net',
        'FPN',
        'PAN',
        'LinkNet',
    ]

    fig = plt.figure(figsize=(5,18))
    
    col=1
    row=7

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