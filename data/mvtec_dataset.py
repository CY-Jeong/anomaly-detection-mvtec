from data.base_dataset import BaseDataset
from data.image_folder import get_datapaths, get_transform
from PIL import Image
import os


class MvtecDataset(BaseDataset):
    """This dataset is given by https://www.mvtec.com/company/research/datasets/mvtec-ad
    THis dataset structure : train/good/[imgs] | test/[categories]/[imgs]
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.img_size
        self.train_dir = os.path.join(opt.data_dir, opt.object, opt.mode)
        self.train_paths = get_datapaths(self.train_dir)
        self.train_size = len(self.train_paths)
        self.transform = get_transform(opt)


    def __getitem__(self, index):
        img_path = self.train_paths[index % self.train_size]
        label = img_path.split('/')[-2]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return {'label' : label,'img' : img, 'path' : img_path}
    def __len__(self):
        return self.train_size