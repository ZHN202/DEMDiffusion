import cv2
import os
import torch
import numpy as np
import matplotlib
import PIL.Image as Image
def colorize_depth_maps(
        depth_map, min_depth, max_depth, cmap="Spectral"
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1) # value from 0 to 1
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


if __name__ == '__main__':
    dem_file = r'G:\Moon1129\testB\512_2560.tif'
    dem = Image.open(dem_file,)
    dem =torch.from_numpy( np.array(dem))
    print(dem.shape)

    min_depth = -5
    max_depth = 10

    # show the image
    img = colorize_depth_maps(dem, min_depth, max_depth)
    img = img.squeeze().numpy()
    img = np.moveaxis(img, 0, -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)



