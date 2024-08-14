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

#############################
# get parameters
#############################
hparams = get_param()


#############################
# set device param
#############################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(114514)
# np.random.seed(114514)


#############################
# training parameters
#############################
N = len(NeRF_train_Dataset(hparams.num_train))
Batch_size = hparams.batch_size
iterations = N // Batch_size

bound = (hparams.lower_bound, hparams.upper_bound)
N_samples = (hparams.rough_samples, hparams.fine_samples)
epoch = hparams.num_epochs
use_view = hparams.use_view


print(f'Each batch has {Batch_size} rays, {iterations} batches in one epoch.')


#############################
# test data
#############################
dataloader = Dataloader.get_test_loader()
dataiter = iter(dataloader)
test_data = next(dataiter)
test_rays_o, test_rays_d, test_rgb = test_data

test_rays_o = torch.squeeze(test_rays_o, dim=0).to(device)
test_rays_d = torch.squeeze(test_rays_d, dim=0).to(device)
test_rgb = torch.squeeze(test_rgb, dim=0).to(device)


#############################
# training
#############################
model = NeRF(use_view_dirs=use_view).to(device)
optimizer = torch.optim.Adam(model.parameters(), hparams.lr)
mse = torch.nn.MSELoss()


for e in range(epoch):
    # create iteration
    dataloader = Dataloader.get_train_loader()
    train_iter = iter(dataloader)

    with tqdm(total=iterations, desc=f"Epoch {e+1}", ncols=100) as p_bar:
        for i in range(iterations):
            train_rays = next(train_iter)
            train_rays = train_rays.to(device)
            assert train_rays.shape == (Batch_size, 9)

            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb, _, __ = render_rays(model, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)

            loss = mse(rgb, target_rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            p_bar.update(1)

    with torch.inference_mode():
        rgb_list = []

        for j in range(test_rays_o.shape[0]):
            rays_od = (test_rays_o[j], test_rays_d[j])
            rgb, _, __ = render_rays(model, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)
            rgb_list.append(rgb.unsqueeze(0))
        rgb = torch.cat(rgb_list, dim=0)

        loss = mse(rgb, test_rgb).cpu()
        psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
        print(f'PSNR={psnr.item()}')

        plt_show(rgb.cpu().numpy(), psnr.numpy(), e, block=False)

        checkpoint = {
            'epoch': e + 1,
            'num_train': 200,
            'ray_sample_rough': 64,
            'ray_sample_fine': 128,
            'lower_bound': 2.,
            'upper_bound': 6.,
            'batch_size': 4096,
            'psnr': psnr.item(),
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }

        FILE = './model/ship_temp_' + str(e) + '.pth'
        torch.save(checkpoint, FILE)


checkpoint = {
    'epoch': 20,
    'num_train': 200,
    'ray_sample_rough': 64,
    'ray_sample_fine': 128,
    'lower_bound': 2.,
    'upper_bound': 6.,
    'batch_size': 4096,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

FILE = './model/ship_400_400.pth'
torch.save(checkpoint, FILE)
