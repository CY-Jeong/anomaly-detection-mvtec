from .base_model import BaseModel
from . import networks
import torch
from utils import utils
from models import init_net
import os

class CAE(BaseModel):
    """This class implements the Convolutional AutoEncoder for normal image generation
    CAE is processed in encoder and decoder that is composed CNN layers
    """

    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        """Initialize the CAE model"""
        BaseModel.__init__(self, opt)
        self.opt = opt
        img_size = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent
        self.encoder = init_net(networks.Encoder(latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize encoder networks doing data parallel and init_weights
        self.decoder = init_net(networks.Decoder(img_size, latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize decoder networks doing data parallel and init_weights
        self.networks = ['encoder', 'decoder']
        self.criterion = torch.nn.MSELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['loss']

        if self.opt.mode == 'train':# if mode is train, we have to set optimizer and requires grad is true
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.set_requires_grad(self.decoder, self.encoder, requires_grad=True)

    def forward(self):
        features = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(features)

    def backward(self):
        self.loss = self.criterion(10*self.real_imgs, 10*self.generated_imgs)
        self.loss.backward()

    def set_input(self, input):
        self.real_imgs = input['img'].to(self.device)

    def train(self):
        self.forward()
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        self.backward()
        self.optimizer_d.step()
        self.optimizer_e.step()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_images(self, data):
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object)
        paths = os.path.join(paths, "result")
        anomaly_img = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img, paths, data)




