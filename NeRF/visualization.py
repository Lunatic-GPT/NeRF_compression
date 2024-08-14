import numpy as np
import matplotlib.pyplot as plt

psnr_list = []
num_epoch = []


# def noraml_01(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))


def plt_show(rgb, psnr, epoch, block=True):
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    plt.imshow(rgb)
    plt.title(f'Epoch: {epoch + 1}')
    # plt.title(f'PSNR: {psnr}')
    # plt.savefig(rgb.cpu().detach().numpy())
    plt.axis('off')

    if block:
        plt.show()
    else:
        plt.ion()
        plt.pause(1)
        plt.close('all')



    #############################
    # fft
    #############################
    # img_fft = np.fft.fft2(rgb)
    # img_shift = np.fft.fftshift(img_fft)
    # # magnitude_spectrum = 20 * np.log(np.abs(img_shift))
    #
    # plt.subplot(122)
    # plt.imshow(np.abs(img_shift), cmap='gray')
    # plt.title(f'Epoch: {epoch + 1}')
    # # plt.savefig(rgb.cpu().detach().numpy())
    # plt.axis('off')





