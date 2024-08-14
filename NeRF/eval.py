import cv2
import torch
import Dataloader

from Model import NeRF
from Render import render_rays

from param import get_param
from visualization import plt_show


hparams = get_param()


bound = (hparams.lower_bound, hparams.upper_bound)
N_samples = (hparams.rough_samples, hparams.fine_samples)
epoch = hparams.num_epochs
use_view = hparams.use_view


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(114514)

dataloader = Dataloader.get_test_loader()
dataiter = iter(dataloader)
test_data = next(dataiter)
test_rays_o, test_rays_d, test_rgb = test_data

test_rays_o = torch.squeeze(test_rays_o, dim=0).to(device)
test_rays_d = torch.squeeze(test_rays_d, dim=0).to(device)
test_rgb = torch.squeeze(test_rgb, dim=0).to(device)


model = NeRF(use_view_dirs=use_view).to(device)
checkpoint = torch.load(hparams.test_model)
# print(checkpoint['psnr'], checkpoint['epoch'])
model.load_state_dict(checkpoint['model_state'])
# model.load_state_dict(checkpoint)

# # create a quantized model instance
# model_int8 = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear},  # a set of layers to dynamically quantize
#     dtype=torch.qint8)  # the target dtype for quantized weights
#
# model = model_int8


mse = torch.nn.MSELoss()
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

    plt_show(rgb.cpu().numpy(), psnr.numpy(), 0, block=True)

    rgb = rgb.cpu().numpy() * 255
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('original_lego.png', img_bgr)


