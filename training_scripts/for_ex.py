import cv2

import PIL.Image as Image

import numpy as np
import matplotlib.pyplot as plt


file_path = r"D:\。\周浩男\毕业设计\Pictures\2304_1536_2.531_pred_vae.tif"
image = Image.open(file_path)
image = np.array(image)
print(image.shape)
plt.plot()
plt.imshow(image, cmap='Spectral_r', vmin=-5, vmax=10)
plt.axis('off')
plt.colorbar()
plt.show()

