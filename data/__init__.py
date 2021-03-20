"""This package include all things related to data.

Image processing, load dataset, make dataloader etc....
"""


from data.base_dataset import BaseDataset
from data.custom_dataloader import CustomDatasetDataLoader

import importlib



def load_dataset(dataset_name):
    """Loading a dataset using dataset_name
    the dataset name's format is [dataset_name]_dataset.py
    """
    dataset_filename = f"data.{dataset_name}_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("Dataset is not existed ")
    return dataset


def create_dataset(opt):
    """Create dataset with dataloader."""
    dataset = load_dataset(opt.dataset)
    dataset = dataset(opt)
    data_loader = CustomDatasetDataLoader(opt, dataset)
    print("dataset [%s] was created" % type(dataset).__name__)
    dataset = data_loader.load_data()
    return dataset




