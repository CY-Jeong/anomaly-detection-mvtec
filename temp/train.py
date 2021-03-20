import yaml
import argparse
import numpy as np

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers


import CAE
import CAExperiment

parser = argparse.ArgumentParser(description='Generic runner for anoGAN model')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False


tb_logger = pl_loggers.TensorBoardLogger('logs/')

model = CAE(config)
experiment = CAExperiment(model, config)

runner = Trainer()


print("-----------TRANING START------------")
runner.fit(experiment)
print("-----------TRANING FINISHING------------")
