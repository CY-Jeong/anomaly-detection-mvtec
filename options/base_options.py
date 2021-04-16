import argparse
import torch
import os
from utils import utils

class BaseOptions():
    """
    This class is for defining arguments during both training and testing.
    Also this class can give some functions. (printing options, parsing)
    """

    def initialize(self, parser):
        """Defining arguments used in both training and testing"""
        parser.add_argument('--data_dir', type=str, default = '/data/cyj/mvtec', help='path to dataset')
        parser.add_argument('--gpu', type=str, default='0', help='gpu number : e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--save_dir', type=str, default='/data/cyj/mvtec/', help='models are saved here')
        parser.add_argument('--model', type=str, default='aae', help='choose which model to use')
        parser.add_argument('--channels', type=int, default=3, help='# of image channels 3:RGB, 1:gray-scale')
        parser.add_argument('--img_size', type=int, default=256, help='img size of input and output for networks')
        parser.add_argument('--latent', type=int, default=100, help='the letent vector size for networks')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--dataset', type=str, default='mvtec', help='chooses how datasets are loaded.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--cropsize', type=int, default=256, help='crop image size')
        parser.add_argument('--object', type=str, default='metal_nut', help='the object for training')
        return parser

    def parse(self):
        """Parse base options and call function printing option .
           If # gpu is more than 0, make gpu ids list"""
        parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt = parser.parse_args()

        self.print_options(opt)

        gpu_ids = opt.gpu.split(',')
        opt.gpu = []
        for id in gpu_ids:
            id = int(id)
            if id >= 0:
                opt.gpu.append(id)
        if len(opt.gpu) > 0:
            torch.cuda.set_device(opt.gpu[0])

        return opt

    def print_options(self, opt):
        """print options and open save folder for saving options
           It will be saved in save_dir/model_name/[mode]opt.txt"""
        message = '----------------------Arguments-------------------------\n'
        for k, v in vars(opt).items():
            message += f'{k:>25}: {v:<30}\n'
        message += '---------------------End--------------------------------\n'
        print(message)

        # saving options
        result_dir = os.path.join(opt.save_dir, opt.model)
        utils.mkdirs(result_dir)
        opt_file_name = os.path.join(result_dir, f'{opt.mode}opt.txt')
        with open(opt_file_name, 'wt') as f:
            f.write(message)
