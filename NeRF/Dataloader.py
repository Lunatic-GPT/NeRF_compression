import os
import cv2
import torch
import numpy as np

from param import get_param
from Render import sample_rays_np
from load_blender import read_pose_blender, read_img_blender

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

hparams = get_param()

path_dir_imgs = hparams.root_dir + '/rgb'
path_dir_poses = hparams.root_dir + '/pose'


class NeRF_train_Dataset(Dataset):
    def __init__(self, num_train):
        self.num_train = num_train

        if hparams.use_tiny_nerf:
            self.data = np.load(hparams.root_dir)

            self.images = self.data['images']   # (106, 100, 100, 3)
            self.poses = self.data['poses']     # (106, 4, 4)
            self.focal = self.data['focal']  # * 4    # one float number 138.88887889922103

            self.H, self.W = self.images.shape[1:3]

            self.images = self.images[0:num_train]
            self.poses = self.poses[0:num_train]

        else:
            self.images = read_img_blender(path_dir_imgs)
            self.poses = read_pose_blender(path_dir_poses)
            self.focal = 138.88887889922103 * 4  # 400 * 400

            self.H, self.W = self.images.shape[1:3]

            self.images = self.images[0:num_train]
            self.poses = self.poses[0:num_train]


        rays_o_list = []
        rays_d_list = []
        rays_rgb_list = []

        for i in range(self.num_train - 0):
            pose = self.poses[i]
            img = self.images[i]
            rays_o, rays_d = sample_rays_np(self.H, self.W, self.focal, pose)

            rays_o_list.append(rays_o.reshape(-1, 3))
            rays_d_list.append(rays_d.reshape(-1, 3))
            rays_rgb_list.append(img.reshape(-1, 3))

        rays_o_npy = np.concatenate(rays_o_list, axis=0)
        rays_d_npy = np.concatenate(rays_d_list, axis=0)
        rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)

        self.rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1))

    def __getitem__(self, idx):

        return self.rays[idx]

    def __len__(self):
        return self.rays.shape[0]


def get_train_loader():
    train_loader = DataLoader(dataset=NeRF_train_Dataset(hparams.num_train),
                              shuffle=True,
                              num_workers=36,
                              batch_size=hparams.batch_size,
                              pin_memory=True)
    return train_loader


# dataloader = get_train_loader()
# dataiter = iter(dataloader)
# data = next(dataiter)
# print(len(NeRF_train_Dataset(hparams.num_train)))


class NeRF_test_Dataset(Dataset):
    def __init__(self, test_idx):

        if hparams.use_tiny_nerf:
            self.data = np.load(hparams.root_dir)
            self.test_idx = test_idx

            self.images = self.data['images']   # (106, 100, 100, 3)
            self.poses = self.data['poses']     # (106, 4, 4)
            self.focal = self.data['focal']  # * 4    # one float number 138.88887889922103

            self.H, self.W = self.images.shape[1:3]

            self.image = self.images[self.test_idx]
            self.pose = self.poses[self.test_idx]

        else:
            self.test_idx = test_idx

            self.images = read_img_blender(path_dir_imgs)
            self.poses = read_pose_blender(path_dir_poses)
            self.focal = 138.88887889922103 * 4  # 400 * 400

            self.H, self.W = self.images.shape[1:3]

            self.image = self.images[self.test_idx]
            self.pose = self.poses[self.test_idx]


        test_rays_o, test_rays_d = sample_rays_np(self.H, self.W, self.focal, self.pose)

        self.test_rays_o = torch.tensor(test_rays_o)
        self.test_rays_d = torch.tensor(test_rays_d)
        self.test_rgb = torch.tensor(self.image)

    def __getitem__(self, item):
        return self.test_rays_o, self.test_rays_d, self.test_rgb

    def __len__(self):
        return 1


def get_test_loader():
    test_loader = DataLoader(dataset=NeRF_test_Dataset(hparams.test_idx),
                             shuffle=False,
                             num_workers=36,
                             batch_size=1,
                             pin_memory=True)
    return test_loader


# dataloader = get_test_loader()
# dataiter = iter(dataloader)
# data = next(dataiter)
# data = torch.squeeze(data[2], dim=0).numpy()
#
# plt.imshow(data)
# plt.title('original')
# # plt.savefig(rgb.cpu().detach().numpy())
# plt.axis('off')
#
# plt.show()
