import torchvision
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np

def imshow_grid(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img_numpy = img.numpy()*0.5+0.5
    print(img_numpy.shape)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()

def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    ## Utils to handle newer PyTorch Lightning changes from version 0.6
    ## ==================================================================================================== ##

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


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