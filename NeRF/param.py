import argparse


def get_param():
    parser = argparse.ArgumentParser()

    #############################
    # dataset parameters
    #############################
    # parser.add_argument('--root_dir', type=str, default='../Lego.npz',
    #                     help='root directory of dataset')

    parser.add_argument('--use_tiny_nerf', type=bool, default=False,  # use_tiny_nerf True, trex False.
                        help='whether to use the LEGO.npz dataset')

    # parser.add_argument('--root_dir', type=str, default='../trex',
    #                     help='root directory of dataset')

    parser.add_argument('--root_dir', type=str, default='../Synthetic_NeRF/Ship',
                        help='root directory of dataset')

    #############################
    # model parameters
    #############################
    parser.add_argument('--fine_samples', type=int, default=128,
                        help='number of each ray fine samples')

    parser.add_argument('--rough_samples', type=int, default=64,
                        help='number of each ray rough samples')

    parser.add_argument('--lower_bound', type=float, default=2.,  # use_tiny_nerf 2., trex 10.
                        help='ray sampling lower bound along z axis')

    parser.add_argument('--upper_bound', type=float, default=6.,  # use_tiny_nerf 6., trex 130.
                        help='ray sampling upper bound along z axis')

    #############################
    # training options
    #############################
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='number of rays in a batch')

    parser.add_argument('--num_train', type=int, default=10,   # use_tiny_nerf 100, trex 50
                        help='number of train data in the whole dataset')

    parser.add_argument('--num_epochs', type=int, default=20,   # use_tiny_nerf 10, trex 200
                        help='number of training epochs')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')

    #############################
    # testing options
    #############################
    parser.add_argument('--use_view', type=bool, default=True,
                        help='whether to use the test img view')

    parser.add_argument('--test_idx', type=int, default=237,  # use_tiny_nerf 74 or 101, trex 53, lego 237
                        help='test image idx')

    parser.add_argument('--test_model', type=str, default='./model/ship_temp_15.pth',
                        help='path to the pretrain model')

    return parser.parse_args()

