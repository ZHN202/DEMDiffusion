import cv2

import PIL.Image as Image

import numpy as np
import rasterio


src_ds = rasterio.open(r"G:\Moon1129\testB\1024_4608.tif")

if src_ds is not None:
    print ( str(src_ds.transform))