import pytorch_lightning as pl
from utils import data_loader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch

from collections import OrderedDict
from .types_ import *


class ENExperiment(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        self.batch_size = config["batch_size"]
        self.img_size = config["batch_size"]
        self.lambda_gp = config["lambda_gp"]
        self.lr = config["lr"]
        self.b1 = config["b1"]
        self.b2 = config["b2"]
        self.object = config["object"]
        self.model = model
        self.model.generator.eval()
        self.model.discriminator.eval()

    def forward(self, input: Tensor, **kwargs):
        _, _, e_imgs = self.model(input, **kwargs)
        return e_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.params["latent_dim"])
        z = z.type_as(imgs)

        lambda_gp = self.lambda_gp

        e_loss = self.model.E_loss_function(imgs, z, lambda_gp)
        tqdm_dict = {'e_loss': e_loss}
        output = OrderedDict({
            'loss': e_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output


    def configure_optimizer(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(b1, b2))

        return [opt_e], []

    @data_loader
    def train_dataloader(self):
        transforms = self.data_transforms()

        train_dataset, _ = ImageFolder('./Downloads/' + self.object + '/train', transform=transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_dataloader

    def data_transforms(self):
        transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size), interpolation=2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return transform

