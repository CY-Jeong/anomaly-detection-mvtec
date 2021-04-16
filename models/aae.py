from .base_model import BaseModel
from . import networks
import torch
from utils import utils
from models import init_net
import os

class AAE(BaseModel):

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
        self.encoder = init_net(networks.Encoder_aae(latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize encoder networks doing data parallel and init_weights
        self.decoder = init_net(networks.Decoder(img_size, latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize decoder networks doing data parallel and init_weights
        self.discriminator = init_net(networks.Discriminator(latent).cuda(), gpu=opt.gpu, mode=opt.mode)
        # initialize discriminator networks doing data parallel and init_weights
        self.networks = ['encoder', 'decoder', 'discriminator']
        self.criterion = torch.nn.MSELoss()
        self.criterion_dm = torch.nn.BCELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['recon_loss', 'dm_loss', 'g_loss']
        self.real_label = torch.ones([self.opt.batch_size, 1])
        self.fake_label = torch.zeros([self.opt.batch_size, 1])
        if self.opt.mode == 'train':# if mode is train, we have to set optimizer and requires grad is true
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_dm = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr/5,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_dm)
            self.set_requires_grad(self.decoder, self.encoder, self.discriminator, requires_grad=True)

    def forward_recon(self):
        z = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(z)

    def forward_dm(self):
        z_fake = self.encoder(self.real_imgs)
        self.fake = self.discriminator(z_fake)
        z_real_gauss = torch.randn(self.real_imgs.size()[0], self.opt.latent)
        self.real = self.discriminator(z_real_gauss)

        self.real_label = self.real_label.type_as(self.real)
        self.fake_label = self.fake_label.type_as(self.fake)

    def backward_recon(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.eval()
        self.recon_loss = self.criterion(10.*self.real_imgs, 10.*self.generated_imgs)
        self.recon_loss.backward()

    def backward_dm(self):
        # discriminator train
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()
        self.dm_loss = self.criterion_dm(self.real, self.real_label) + self.criterion_dm(self.fake, self.fake_label)
        self.dm_loss.backward()

    def backward_g(self):
        # generator train
        self.encoder.train()
        self.discriminator.eval()
        self.fake = self.discriminator(self.encoder(self.real_imgs))
        self.g_loss = self.criterion_dm(self.fake, self.real_label)
        self.g_loss.backward()

    def set_input(self, input):
        self.real_imgs = input['img'].to(self.device)

    def train(self):
        # recon train
        self.forward_recon()
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        self.backward_recon()
        self.optimizer_d.step()
        self.optimizer_e.step()

        # discriminator train
        self.forward_dm()
        self.optimizer_dm.zero_grad()
        self.backward_dm()
        self.optimizer_dm.step()

        # generator train
        self.optimizer_d.zero_grad()
        self.backward_g()
        self.optimizer_d.step()

    def test(self):
        with torch.no_grad():
            self.forward_recon()

    def save_images(self, data):
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object)
        paths = os.path.join(paths, "result")
        anomaly_img = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img, paths, data)




