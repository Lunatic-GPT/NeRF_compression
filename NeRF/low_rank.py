import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import Dataloader
import matplotlib.pyplot as plt

from tqdm import tqdm

from Model import NeRF
from Render import render_rays

from param import get_param
from Dataloader import NeRF_train_Dataset
from visualization import plt_show


import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import (
    ConstraintL0Pruning,
    LowRank,
    RankSelection,
    AdaptiveQuantization,
)

#############################
# get parameters
#############################
hparams = get_param()


#############################
# set device param
#############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


print(f"Each batch has {Batch_size} rays, {iterations} batches in one epoch.")


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
# load model state
#############################

model = NeRF(use_view_dirs=use_view).to(device)
checkpoint = torch.load(hparams.test_model)
print(checkpoint["psnr"], checkpoint["epoch"])
model.load_state_dict(checkpoint["model_state"])

mse = torch.nn.MSELoss()


#############################
# render
#############################

with torch.inference_mode():
    rgb_list = []

    for j in range(test_rays_o.shape[0]):
        rays_od = (test_rays_o[j], test_rays_d[j])
        rgb, _, __ = render_rays(
            model,
            rays_od,
            bound=bound,
            N_samples=N_samples,
            device=device,
            use_view=use_view,
        )
        rgb_list.append(rgb.unsqueeze(0))
    rgb = torch.cat(rgb_list, dim=0)

    loss = mse(rgb, test_rgb).cpu()
    psnr = -10.0 * torch.log(loss).item() / torch.log(torch.tensor([10.0]))
    print(f"PSNR={psnr.item()}")
    plt.imshow(rgb.cpu().numpy())
    plt.savefig("original")

#############################
# COMPRESSION: l_step function
#############################


def my_l_step(model, lc_penalty, step):
    train_loader = Dataloader.get_train_loader()
    train_iter = iter(train_loader)

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    lr = 0.7 * (0.98**step)
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    print(f"L-step #{step} with lr: {lr:.5f}")
    epochs_per_step_ = 7
    if step == 0:
        epochs_per_step_ = epochs_per_step_ * 2
    avg_loss = []
    with tqdm(total=iterations, desc=f"Epoch", ncols=100) as p_bar:
        for i in range(iterations):
            train_rays = next(train_iter)
            train_rays = train_rays.to(device)
            assert train_rays.shape == (Batch_size, 9)

            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb, _, __ = render_rays(
                model,
                rays_od,
                bound=bound,
                N_samples=N_samples,
                device=device,
                use_view=use_view,
            )

            loss = mse(rgb, target_rgb) + lc_penalty()  # lc_penalty
            avg_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({"loss": "{0:1.5f}".format(loss.item())})
            p_bar.update(1)

    print(f"\tepoch finished.")
    print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")


#############################
# train_test_acc_eval_f
#############################


def train_test_acc_eval_f(net):
    # torch.save(net.state_dict(), './model/low_rank_compressed_200_temp')
    with torch.inference_mode():
        rgb_list = []

        for j in range(test_rays_o.shape[0]):
            rays_od = (test_rays_o[j], test_rays_d[j])
            rgb, _, __ = render_rays(
                net,
                rays_od,
                bound=bound,
                N_samples=N_samples,
                device=device,
                use_view=use_view,
            )
            rgb_list.append(rgb.unsqueeze(0))
        rgb = torch.cat(rgb_list, dim=0)

        loss = mse(rgb, test_rgb).cpu()
        psnr = -10.0 * torch.log(loss).item() / torch.log(torch.tensor([10.0]))
        print(f"PSNR={psnr.item()}")
        plt.imshow(rgb.cpu().numpy())
        plt.savefig("test")
        # plt_show(rgb.cpu().numpy(), psnr.numpy(), 0)


#############################
# compression tasks
#############################

layers = [
    lambda x=x: getattr(x, "weight")
    for x in model.modules()
    if isinstance(x, nn.Linear)
]

for x in model.modules():
    if isinstance(x, nn.Linear):
        print(x)

alpha = 1e-9
compression_tasks = {
    Param(layers[1], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.1.weight",
    ),
    Param(layers[2], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.2.weight",
    ),
    Param(layers[3], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.3.weight",
    ),
    Param(layers[4], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.4.weight",
    ),
    Param(layers[6], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.6.weight",
    ),
    Param(layers[7], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "net.7.weight",
    ),
    Param(layers[9], device): (
        AsIs,
        LowRank(
            target_rank=32
        ),
        "feature_linear.weight",
    ),
}

mu_s = [9e-5 * (1.1**n) for n in range(20)]


lc_alg = lc.Algorithm(
    model=model,  # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,  # implementation of L-step
    mu_schedule=mu_s,  # schedule of mu values
    evaluation_func=train_test_acc_eval_f,  # evaluation function
)


lc_alg.run()
print('Compressed_params:', lc_alg.count_params())

lc_alg.compression_eval()

torch.save(lc_alg.model.state_dict(), './model/low_rank_32_10__0_10_237_lego')
