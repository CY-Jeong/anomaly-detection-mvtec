import torch
from unittest import TestCase, main
from models import VanillaVAE
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import yaml
import CAE
from utils import compare_images


class TestAnoGAN(TestCase):

    def setUp(self) -> None:
        parser = argparse.ArgumentParser(description='Generic Test AnoGAN')
        parser.add_argument('--config', '-c',
                            dest="filename",
                            metavar='FILE',
                            help='path to the config file',
                            default='configs.yaml')
        args = parser.parse_args()
        with open(args.filename, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        self.img_size = config["img_size"]
        self.model = CAE()
        object = config["object"]
        #data
        transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size), interpolation=2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset, _ = ImageFolder('./Downloads/' + object + '/test', transform=transforms)

        self.train_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def test_show_imgs_diff(self):
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        criterion = nn.MSELoss()
        kappa = 1.0
        for i, (img,label) in enumerate(self.train_dataloader):
            a = label.numpy()[0]

            if a != 4:
                continue
            real_img = img.to(self.device)
            real_z, _, _ = self.model(real_img)
            generated_imgs = self.model(x)
            fake_z, _, _ = self.model(generated_imgs)  # torch.Size([1, 100]) --> latent 진짜 이미지와 매핑된 가짜 이미지의 latent vector

            real_feature = D.forward_features(real_img)  # 1, 256
            fake_feature = D.forward_features(generated_imgs)
            real_feature = real_feature / real_feature.max()
            fake_feature = fake_feature / fake_feature.max()

            img_distance = criterion(generated_imgs, real_img)
            loss_feature = criterion(fake_feature, real_feature)

            anomaly_score = img_distance + kappa * loss_feature

            z_distance = criterion(fake_z, real_z)
            with open("score.csv", "a") as f:
                f.write(f"{label.item()},{img_distance}," f"{anomaly_score},{z_distance}\n")

            print(f"{label.item()}, {img_distance}, "
                  f"{anomaly_score}, {z_distance}\n")

            compare_images(real_img, fake_img, i, anomaly_score, reverse=False, threshold=0.2)

if __name__ == '__main__':


    main()