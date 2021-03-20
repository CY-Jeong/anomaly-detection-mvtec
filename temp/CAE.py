from models import Encoder, Decoder
from .types_ import *

import torch
from torch import autograd
import torch.nn as nn

class CAE():
    def __init__(self, config: dict, device):
        channels = config["channels"]
        img_size = config["img_size"]
        latent_dim = config["latent_dim"]
        img_shape = (channels, img_size, img_size)
        self.encoder = Encoder(img_shape, latent_dim)
        self.decoder = Decoder(latent_dim).to(device)

    def forward(self, z):
        z = self.encoder(z)
        fake_imgs = self.decoder(z)
        return z, fake_imgs

    def G_loss_function(self, real_imgs, gp, lambda_gp):
        criterion = nn.MSELoss()
        features = self.encoder(real_imgs)
        generated_imgs = self.decoder(features)
        real_imgs = real_imgs.type_as(features)
        loss = criterion(real_imgs, generated_imgs)

        return loss

    def sample(self, num_samples: int, current_device: int, latent_dim: int):
        z = torch.randn(num_samples, latent_dim)
        z = z.to(current_device)
        _, samples = self.decoder(z)

        return samples

