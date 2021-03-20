"""This code is for detecting anomaly parts in images
There are so many algorithms that help to detect anomaly like mvtecAD, anogan etc..
This scripts show anomaly detection using simple Auto Encoder.

you need to specify the folder which stores datasets(mvtec), and model you gonna use(AE, AAE)

Example:
    test:
        python test.py --datadir [dataset folder] --model [model_name]
"""

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

import time

if __name__ == "__main__":
    opt = TestOptions().parse()     # get test options
    dataset = create_dataset(opt)   # get dataset (mvtec)
    model = create_model(opt)       # create model (AE, AAE)
    dataset_size = len(dataset)

    print(f"Test size is = {dataset_size}")
    model.setup(opt)                # set model : if mode is 'train', define schedulers and if mode is 'test', load saved networks
    model.eval()                    # model eval version
    for i, data in enumerate(dataset):
        epoch_start_time = time.time()
        model.set_input(data)
        model.test()

        generated_images = model.get_generated_imags()

        epoch_time = time.time() - epoch_start_time
        print(f"{i} epoch_time : {epoch_time:.3f}")
        model.save_images(data)
    print("end Test")
