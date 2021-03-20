import pytorch_lightning as pl
from utils import data_loader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch

from collections import OrderedDict
from .types_ import *

class WGExperiment(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        self.batch_size = config["batch_size"]
        self.img_size = config["batch_size"]
        self.lambda_gp = config["lambda_gp"]
        self.lr = config["lr"]
        self.b1 = config["b1"]
        self.b2 = config["b2"]
        self.object = config["object"]
        self.model = model

    def forward(self, input: Tensor, **kwargs):
        _, g_imgs, _ = self.model(input, **kwargs)
        return g_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        g_loss = self.G_loss_function(imgs)
        tqdm_dict = {'g_loss': g_loss}
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def configure_optimizer(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_e = torch.optim.Adam(self.discriminator.parameters(), lr = lr, betas = (b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr = lr, betas = (b1, b2))

        return [opt_g, opt_e], []

    @data_loader
    def train_dataloader(self):
        transforms = self.data_transforms()

        train_dataset, _ = ImageFolder('./Downloads/' + self.object + '/train', transform=transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_dataloader
    # @data_lodaer
    # def val_dataloader(self):
    #     transforms = self.data_transforms()
    #
    #     _, val_dataset = train_val_split()
    #
    #     train_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    def data_transforms(self):
        transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size), interpolation=2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return transform
    # def train_val_split(self):
    #     dataset = ImageFolder('./Downloads/' + object + '/train', transform=transforms)
    #     lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    #     train_set, val_set = random_split(dataset, lengths)
    #     return train_set, val_set
