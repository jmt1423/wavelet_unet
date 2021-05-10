# Notes on the masking process #

======= *2021-05-10 13:08* =======

It seems as if the YellowFlowers1.shp only works for the 10meter_ortho_R1C3.tif
file and the YellowFlowers2.shp only works for the 10meter_ortho_R1C4.tif file.

When using fiona and rasterio to create raster image (mask) from shape files
they both work, however if you switch either the tif or the shapefile the code
cannot find or create a mask for the chosen image.

For a start I have saved each mask to a masks directory inside the drone_data
main directory.

======= *2021-05-10 17:37* =======

As of now, the UNET trains on the single given image and a mask created from the
given shape files. However The predicted results is just an entire screen of
white, so I think we need more data to do any serious training on the model.
