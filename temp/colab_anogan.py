import torch
import torch.nn as nn
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset
from parallel import DataParallelModel, DataParallelCriterion

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import random
import os


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

writer = SummaryWriter(logdir='runs/GAN_training')
#device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3, 5, 6, 7'


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

        def block(in_feature, out_feature, normalize=False):
            layers = [nn.Linear(in_feature, out_feature)]
            # if normalize:
            #     layers.append(nn.InstanceNorm1d(out_feature, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.model(input)

        return img.view(img.shape[0], *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward_features(self, img):
        img_flat = img.view(img.shape[0], -1) # 이미지가 들어올 때 Linear에 들어가기 위해 shape을 Flatten 해줌.
        features = self.features(img_flat)
        return features

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity
class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
n_epochs = 200
batch_size = 8
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 256
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

G = nn.DataParallel(Generator(img_shape, latent_dim), output_device = 5)#.to(device)
D = nn.DataParallel(Discriminator(img_shape), output_device = 5)#.to(device)
G.cuda()
D.cuda()
optimizer_G = optim.Adam(G.parameters(), lr=lr, weight_decay=1e-5)
optimizer_D = optim.Adam(D.parameters(), lr=lr, weight_decay=1e-5)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1).type_as(real_samples)#.to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape).type_as(d_interpolates)#.to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


padding_epoch = len(str(n_epochs))  # 3
padding_i = len(str(len(train_dataloader)))  # 2

d_losses = []
g_losses = []
from torchsummary import summary
summary(G, (1, latent_dim))
summary(D, (1, channels, img_size, img_size))

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        z = torch.randn(imgs.shape[0], latent_dim)#.to(device) # 64, 100
        fake_imgs = G(z)
        real_imgs = imgs.type_as(fake_imgs[0])  # .to(device)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)

        # Adversarial loss
        real_validity = D(real_imgs)
        fake_validity = D(fake_imgs)
        d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty)
        d_loss.backward()

        optimizer_D.step()

        if (i + 1) % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            fake_imgs = G(z)
            fake_validity = D(fake_imgs)

            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            writer.add_scalars('GAN loss', {'G_loss': g_loss.item(), 'D_loss': d_loss.item()}, epoch)

            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                  f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                  f"[D loss: {d_loss.item():3f}] "
                  f"[G loss: {g_loss.item():3f}]")
# writer.close()
G_params = {
    'epoch': n_epochs,
    'model_state_dict': G.state_dict(),
    'optimizer_state_dict': optimizer_G.state_dict(),
    'loss': g_loss
}
D_params = {
    'epoch': n_epochs,
    'model_state_dict': D.state_dict(),
    'optimizer_state_dict': optimizer_D.state_dict(),
    'loss': d_loss
}
# torch.save(G_params, "./saved_models/" + object + "_G.pth")
# torch.save(D_params, "./saved_models/" + object + "_D.pth")
print('Finished Training')


plt.plot(g_losses, label='g_loss')
plt.plot(d_losses, label='d_loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

writer = SummaryWriter(logdir='runs/Encoder_training')
img_shape = (channels, img_size, img_size)

E = nn.DataParallel(Encoder(img_shape, latent_dim), output_device = 3)#.to(device)
E.cuda()
G.eval()
D.eval()

criterion = nn.MSELoss()

optimizer_E = torch.optim.Adam(E.parameters(), lr=lr, betas=(b1, b2))

padding_epoch = len(str(n_epochs))
padding_i = len(str(len(train_dataloader)))
kappa = 1.0
e_losses = []

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):

        optimizer_E.zero_grad()
        z = E(real_imgs)  # 64, 100
        real_imgs = imgs.type_as(z)#.to(device)

        fake_imgs = G(z)

        real_features = D.forward_features(real_imgs)
        fake_features = D.forward_features(fake_imgs)

        # izif architecture
        loss_imgs = criterion(fake_imgs, real_imgs)
        loss_features = criterion(fake_features, real_features)
        e_loss = loss_imgs + kappa * loss_features

        e_loss.backward()
        optimizer_E.step()

        if (i + 1) % n_critic == 0:
            e_losses.append(e_loss)

            writer.add_scalar('e_loss', e_loss.item(), epoch)
            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                  f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                  f"[E loss: {e_loss.item():3f}]")

# writer.close()
E_params = {
    'epoch': n_epochs,
    'model_state_dict': E.state_dict(),
    'optimizer_state_dict': optimizer_E.state_dict(),
    'loss': e_loss
}
torch.save(E_params, "./saved_models/" + object + "_E.pth")
print('Finished Training')

z = torch.randn(4, latent_dim)#.to(device) # 4, 100
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