import torch
import torch.nn as nn
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import os


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

writer = SummaryWriter(logdir='runs/GAN_training')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.cuda.set_device(device)

def imshow_grid(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img_numpy = img.numpy()*0.5+0.5
    print(img_numpy.shape)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            # State (100x1x1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(True),

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(True),

            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(True),

            # State (32x256x256)
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1),

            nn.Tanh())

    def forward(self, input):
        img = self.model(input)

        return img.view(img.shape[0], *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            # State Cx512x512)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (8x256x256)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (16x128x128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4),
            # Output (1024x1x1)
        )



    def forward(self, img):

        validity = F.sigmoid(self.model(img))
        return validity

class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            # State Cx512x512)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (16x256x256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x128x128)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # output of main module --> State (1024x4x4)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=100, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, img):
        features = self.model(img)
        features = self.last_layer(features)
        return features
n_epochs = 100
batch_size = 8
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 512
channels = 3
n_critic = 5
sample_interval = 400
training_label = 0
split_rate = 0.8
lambda_gp = 10
object = "capsule"

# Train Data
from torchvision.datasets import ImageFolder
transform = transform = transforms.Compose([transforms.Resize((img_size,img_size), interpolation=2),
                                            #transforms.Grayscale(num_output_channels=1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = ImageFolder('./Downloads/'+object+'/train', transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

img_shape = (channels, img_size, img_size) # (3, 100, 100)

G = Generator(img_shape, latent_dim).to(device)
D = Discriminator(img_shape).to(device)
E = Encoder(img_shape, latent_dim).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr, weight_decay=1e-5)
optimizer_D = optim.Adam(D.parameters(), lr=lr, weight_decay=1e-5)
optimizer_E = torch.optim.Adam(E.parameters(), lr=lr, betas=(b1, b2))

padding_epoch = len(str(n_epochs))  # 3
padding_i = len(str(len(train_dataloader)))  # 2

summary(G, (latent_dim, 1, 1))
summary(D, (channels, img_size, img_size))
summary(E, (channels, img_size, img_size))
EPS = 0.000001
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        E.eval()
        G.eval()
        D.train()
        #real_imgs = imgs.type_as(fake_imgs[0])  # .to(device)
        imgs = imgs.to(device)
        features = E(imgs).to(device)
        fake_imgs = G(features).to(device)
        # Adversarial loss
        real_validity = D(imgs).to(device)
        fake_validity = D(fake_imgs).to(device)
        d_loss = -torch.mean(torch.log(real_validity+EPS) + torch.log(1-fake_validity+EPS))
        d_loss.backward()

        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        D.eval()
        E.train()
        G.train()
        loss = nn.MSELoss()
        features = E(imgs).to(device)
        fake_imgs = G(features).to(device)
        fake_validity = D(fake_imgs).to(device)

        loss1 = -torch.mean(torch.log(fake_validity))
        loss2 = loss(imgs, fake_imgs)
        g_loss = loss1 + loss2

        g_loss.backward()
        optimizer_G.step()
        optimizer_E.step()

        if (i+1)%5 == 0:
            writer.add_scalars('GAN loss', {'G_loss': g_loss.item(), 'D_loss': d_loss.item()}, epoch)

            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                  f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                  f"[D loss: {d_loss.item():3f}]"
                  f"[G loss: {g_loss.item():3f}]"
                  f"[loss1 : {loss1:3f}]"
                  f"[loss2 : {loss2:3f}]")

print('Finished Training')

writer = SummaryWriter(logdir='runs/Encoder_training')
img_shape = (channels, img_size, img_size)

optimizer_E = torch.optim.Adam(E.parameters(), lr=lr, betas=(b1, b2))

padding_epoch = len(str(n_epochs))
padding_i = len(str(len(train_dataloader)))
kappa = 1.0


# torch.save(E_params, "./saved_models/" + object + "_E.pth")
print('Finished Training')

z = torch.randn(2, latent_dim, 1, 1)#.to(device) # 4, 100
fake_imgs = G(z) # batch size, 3, 100, 100
fake_z = E(fake_imgs) #4, 100
reconfiguration_imgs = G(fake_z)

imshow_grid(reconfiguration_imgs)

# Test Data
from torchvision.datasets import ImageFolder
transform = transform = transforms.Compose([transforms.Resize((img_size,img_size), interpolation=2),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = ImageFolder('./Downloads/'+object+'/test', transform = transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


def compare_images(real_img, generated_img, i, score, reverse=False, threshold=0.1):
    print(real_img.shape)
    real_img = np.transpose(real_img.cpu().detach().numpy()[0], (1, 2, 0)) * 0.5 + 0.5
    generated_img = np.transpose(generated_img.cpu().detach().numpy()[0], (1, 2, 0)) * 0.5 + 0.5

    negative = np.zeros_like(real_img)

    if not reverse:
        diff_img = np.abs(real_img - generated_img)
    else:
        diff_img = np.abs(generated_img - real_img)
    a = np.amax(np.amax(diff_img, axis=0), axis=0)
    b = np.amin(np.amin(diff_img, axis=0), axis=0)
    print(f"amax : {a}, amin : {b}")
    diff_img[diff_img <= threshold] = 0

    anomaly_img = np.zeros_like(real_img)
    anomaly_img[:, :, :] = real_img
    anomaly_img[:, :, 0] = anomaly_img[:, :, 0] + 10. * np.mean(diff_img, axis=2)
    # anomaly_img = anomaly_img.astype(np.uint8)

    fig, plots = plt.subplots(1, 4)
    fig.suptitle(f'Anomaly - (anomaly score: {score:.4})')

    fig.set_figwidth(9)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(real_img, label='real')
    plots[1].imshow(generated_img)
    plots[2].imshow(diff_img)
    plots[3].imshow(anomaly_img)

    plots[0].set_title('real')
    plots[1].set_title('generated')
    plots[2].set_title('difference')
    plots[3].set_title('Anomaly Detection')
    plt.show()

    img_shape = (channels, img_size, img_size)

    criterion = nn.MSELoss()
    G.eval()
    D.eval()
    E.eval()

    with open("score.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance\n")

    for i, (img, label) in enumerate(test_dataloader):
        a = label.numpy()[0]

        if a != 4:
            continue
        real_z = E(real_img)  # 진짜 이미지의 latent vector
        fake_img = G(real_z)  # G에 넣어서 가짜 이미지 생성.
        fake_z = E(fake_img)  # torch.Size([1, 100]) --> latent 진짜 이미지와 매핑된 가짜 이미지의 latent vector
        real_img = img.type_as(real_z)  # .to(device)
        real_feature = D.forward_features(real_img)  # 1, 256
        fake_feature = D.forward_features(fake_img)
        real_feature = real_feature / real_feature.max()
        fake_feature = fake_feature / fake_feature.max()

        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)

        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)
        with open("score.csv", "a") as f:
            f.write(f"{label.item()},{img_distance}," f"{anomaly_score},{z_distance}\n")

        print(f"{label.item()}, {img_distance}, "
              f"{anomaly_score}, {z_distance}\n")

        compare_images(real_img, fake_img, i, anomaly_score, reverse=False, threshold=0.2)


img_shape = (channels, img_size, img_size)

criterion = nn.MSELoss()
G.eval()
D.eval()
E.eval()

for i, (img, label) in enumerate(test_dataloader):
    a = label.numpy()[0]

    if a != 1:
        continue
    real_img = img.to(device)

    real_z = E(real_img)  # 진짜 이미지의 latent vector
    fake_img = G(real_z)  # G에 넣어서 가짜 이미지 생성.
    fake_z = E(fake_img)  # torch.Size([1, 100]) --> latent 진짜 이미지와 매핑된 가짜 이미지의 latent vector

    real_feature = D.forward_features(real_img)  # 1, 256
    fake_feature = D.forward_features(fake_img)
    real_feature = real_feature / real_feature.max()
    fake_feature = fake_feature / fake_feature.max()

    img_distance = criterion(fake_img, real_img)
    loss_feature = criterion(fake_feature, real_feature)

    anomaly_score = img_distance + kappa * loss_feature

    z_distance = criterion(fake_z, real_z)
    with open("score.csv", "a") as f:
        f.write(f"{label.item()},{img_distance}," f"{anomaly_score},{z_distance}\n")

    print(f"{label.item()}, {img_distance}, "
          f"{anomaly_score}, {z_distance}\n")

    compare_images(real_img, fake_img, i, anomaly_score, reverse=False, threshold=0.40)