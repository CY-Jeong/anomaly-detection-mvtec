from abc import ABC, abstractmethod
from . import get_scheduler
from utils import utils

import os
import torch
from collections import OrderedDict

class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.gpu = opt.gpu
        self.device = torch.device(f'cuda:{self.gpu[0]}') if self.gpu else torch.device('cpu')
        self.optimizers = []
        self.networks = []
        self.save_dir = os.path.join(opt.save_dir, opt.object)
        if self.opt.mode == 'Train':
            self.isTrain = True
        elif self.opt.mode == 'Pretrained' or self.opt.mode == 'Test':
            self.isTrain = False
    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def test(self):
        pass

    def setup(self, opt):
        if opt.mode == 'train':
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        elif opt.mode == 'test':
            self.load_networks()
        self.print_networks(opt.verbose)

    def set_requires_grad(self, *nets, requires_grad=False):
        for _, net in enumerate(nets):
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_generated_imags(self):
        visual_imgs = None
        for name in self.visual_names:
            if isinstance(name, str):
                visual_imgs = getattr(self, name)
        return visual_imgs

    def eval(self):
        for name in self.networks:
            net = getattr(self, name)
            net.eval()

    def update_learning_rate(self, epoch):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'{epoch} : learning rate {old_lr:.7f} -> {lr:.7f}')
    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.networks:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self):
        utils.mkdirs(self.save_dir)
        save_encoder_filename = f'{self.model_name}_e.pth'
        save_decoder_filename = f'{self.model_name}_d.pth'
        save_encoder_path = os.path.join(self.save_dir, save_encoder_filename)
        save_decoder_path = os.path.join(self.save_dir, save_decoder_filename)
        net_d = getattr(self, 'decoder')
        net_e = getattr(self, 'encoder')

        if len(self.gpu) > 0 and torch.cuda.is_available():
            torch.save(net_d.module.cpu().state_dict(), save_decoder_path)
            net_d.cuda(self.gpu[0])
            torch.save(net_e.module.cpu().state_dict(), save_encoder_path)
            net_e.cuda(self.gpu[0])
        else:
            torch.save(net_d.cpu().state_dict(), save_decoder_path)
            torch.save(net_e.cpu().state_dict(), save_encoder_path)

    def load_networks(self):
        load_encoder_filename = f'{self.model_name}_e.pth'
        load_decoder_filename = f'{self.model_name}_d.pth'
        load_encoder_path = os.path.join(self.save_dir, load_encoder_filename)
        load_decoder_path = os.path.join(self.save_dir, load_decoder_filename)
        net_e = getattr(self, 'encoder')
        net_d = getattr(self, 'decoder')
        if isinstance(net_d, torch.nn.DataParallel):
            net_d = net_d.module
        if isinstance(net_e, torch.nn.DataParallel):
            net_e = net_e.module
        print('loading the encoder from %s' % load_encoder_path)
        print('loading the decoder from %s' % load_decoder_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        encoder_state_dict = torch.load(load_encoder_path)
        decoder_state_dict = torch.load(load_decoder_path)

        net_e.load_state_dict(encoder_state_dict)
        net_d.load_state_dict(decoder_state_dict)


    def get_current_losses(self, *loss_name):
        loss = {}
        for name in loss_name:
            loss[name] = (float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
        return loss



