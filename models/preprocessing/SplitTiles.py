import os
from itertools import product
import rasterio as rio
from rasterio import windows
import time
from PIL import Image

in_path = '/home/jont/Documents/CoastSat/data/blackpool/images/'
input_filename = 'bp_resized.jpg'

out_path = '/home/jont/Documents/CoastSat/data/blackpool/images/image_tiles/'
output_filename = 'tile_{}-{}.jpg'


def get_tiles(ds, width=722, height=25):
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
    #tile_width, tile_height = 512, 512

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
