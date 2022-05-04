from PIL import Image
import numpy as np
import os

def rename_files():
    mask_output = '../../../masks_final/'
    image_output = '../../../images_final/'

    mask_input = '../../../masks_extended/'
    image_input = '../../../images/'

    num = 0

    for image in os.listdir(image_input):
        im = Image.open(image_input+image)
        im.save(image_output+'newtile_'+str(num)+'.tif')
        num += 1

if __name__ == '__main__':
    rename_files()