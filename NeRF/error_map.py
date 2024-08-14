



import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


def standardization(data):#标准化变成-1到1
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def normalization(data):#归一化变成0-1
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


img1 = cv2.imread('./results/3bit_10__0_10_237_lego_original.png')
img2 = cv2.imread('./results/3bit_10__0_10_237_lego_test.png')

img_error = img2 - img1

img_error = cv2.applyColorMap(img_error, 2)
cv2.imshow('image', img_error)
cv2.waitKey(0)
