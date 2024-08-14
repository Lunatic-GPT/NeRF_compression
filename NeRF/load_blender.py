import os
import cv2
import torch
import numpy as np
from PIL import Image

from param import get_param
from Render import sample_rays_np

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

hparams = get_param()

# path_dir_imgs = '../Synthetic_NeRF/Lego/rgb'
# path_dir_poses = '../Synthetic_NeRF/Lego/pose'


def pose_transform(pose):
    rot = np.array([[1, -1, -1, 1],
                    [1, -1, -1, 1],
                    [1, -1, -1, 1],
                    [1,  1,  1, 1]], dtype=np.float32)
    return rot * pose


# def normalization(data):
#     range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / range


def read_img_blender(path_dir):
    imgs = []
    imgs_list = sorted(os.listdir(path_dir))
    for filename in imgs_list:
        img = Image.open(os.path.join(path_dir, filename))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        img_tmp = Image.new('RGB', size=(800, 800), color=(255, 255, 255))  # white background
        img_tmp.paste(img, (0, 0), mask=img)

        img = np.asarray(img_tmp, dtype=np.float32) / 255.
        # img = normalization(img)
        img = cv2.resize(img, (400, 400))

        imgs.append(img)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
    return np.asarray(imgs)  # (B, H, W, 3)


def read_pose_blender(path_dir):
    poses = []
    poses_list = sorted(os.listdir(path_dir))
    for filename in poses_list:
        pose_tmp = np.loadtxt(os.path.join(path_dir, filename), dtype=np.float32)
        pose_tmp = pose_transform(pose_tmp)
        poses.append(pose_tmp)

    return np.asarray(poses)  # (B, 4, 4)


# lego = read_img_blender(path_dir_imgs)
# print(lego.dtype)

# pose = read_pose_blender(path_dir_poses)
# print(pose[1])
#
#
# data = np.load(hparams.root_dir)
# images = data['images']  # (106, 100, 100, 3)
# poses = data['poses']  # (106, 4, 4)
# focal = data['focal']  # * 4    # one float number 138.88887889922103
# plt.imshow(images[0])
# plt.show()
# print(poses[1])