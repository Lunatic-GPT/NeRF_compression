import torch
import numpy as np
import Dataloader
import matplotlib.pyplot as plt

from tqdm import tqdm

from Model import NeRF
from Render import render_rays

from param import get_param
from Dataloader import NeRF_train_Dataset
from visualization import plt_show

hparams = get_param()

use_view = hparams.use_view
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Model_size：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size


model = NeRF(use_view_dirs=use_view).to(device)
checkpoint = torch.load(hparams.test_model)
# print(checkpoint['psnr'], checkpoint['epoch'])
model.load_state_dict(checkpoint['model_state'])
# model.load_state_dict(checkpoint)


param_size, param_sum, buffer_size, buffer_sum, all_size = getModelSize(model)

print(param_size, param_sum, buffer_size, buffer_sum, all_size)