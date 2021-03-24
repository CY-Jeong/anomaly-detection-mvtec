import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class for datasets"""
    def __init__(self, opt):
        self.args = opt
        self.root = opt.data_dir

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


