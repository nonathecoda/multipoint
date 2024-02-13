# load five images resize them to thermal image size and plot them using matplotlib
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

path_T = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT/camT_00196_00119618_rect.tiff"
path_1 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1/cam1_00203_e_0012_g_01_119881749_corr_rect.tiff"
path_2 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam2/cam2_00203_e_0012_g_01_119881749_corr_rect.tiff"
path_3 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam3/cam3_00203_e_0012_g_01_119881749_corr_rect.tiff"
path_4 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam4/cam4_00203_e_0012_g_01_119881749_corr_rect.tiff"

img_T = cv2.imread(path_T, 0)
img_1 = cv2.imread(path_1, 0)
img_2 = cv2.imread(path_2, 0)
img_3 = cv2.imread(path_3, 0)
img_4 = cv2.imread(path_4, 0)

#resize

img_1 = cv2.resize(img_1, (img_T.shape[1], img_T.shape[0]))
img_2 = cv2.resize(img_2, (img_T.shape[1], img_T.shape[0]))
img_3 = cv2.resize(img_3, (img_T.shape[1], img_T.shape[0]))
img_4 = cv2.resize(img_4, (img_T.shape[1], img_T.shape[0]))

fig, ax = plt.subplots(1, 5, figsize=(10, 10))
ax[0].imshow(img_T, cmap="gray")
ax[0].set_title("Thermal")
ax[0].axis("off")
ax[1].imshow(img_1, cmap="gray")
ax[1].set_title("Cam1")
ax[1].axis("off")
ax[2].imshow(img_2, cmap="gray")
ax[2].set_title("Cam2")
ax[2].axis("off")
ax[3].imshow(img_3, cmap="gray")
ax[3].set_title("Cam3")
ax[3].axis("off")
ax[4].imshow(img_4, cmap="gray")
ax[4].set_title("Cam4")
ax[4].axis("off")
plt.show()