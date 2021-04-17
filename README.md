# Anomaly-Detection
Pytorch implemetation of **Anomaly Detection** which detect not normal images in industrial datasets [mvtec](https://www.mvtec.com/)<br>
It has only simple layers but if you want to look out mvtec paper click [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf).<br>
Actually [mvtecAD](https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection) didn't work well due to finding automatic threshold in finetune stage in my guess.
And there is another anomaly detection that is called [anoGAN](https://arxiv.org/pdf/1703.05921.pdf).
Anogan use medical images that has 64x64 resolution. And it's too slow because they have to train in testing stage for training z vector.
Even if the [fast-anogan](https://www.researchgate.net/publication/330796048_f-AnoGAN_Fast_Unsupervised_Anomaly_Detection_with_Generative_Adversarial_Networks) came out, they use wgan that use linear layers. Using linear layers make constraints that don't make high resolution, and the constraint make networks can't detect small detected parts.<br>
So I use AutoEncoder and AAE.
I use conv2d layers in both of them and can make high resolution images(256x256x3). And I use MSE loss in AutoEncoder.
But only MSE loss make output images blur. So I make decision to use discriminator to make image clear. That's why I use AAE.
There is one weekness in this code, which is we have to use manualy fixed threshold.

## Tensorflow version
Tensorflow implementation of **[MVTEC-AD](https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection)** which is implemetation in mvtec paper linked above.


## Simple CNN models(AutoEncoder)

<img src='imgs/layers_AE.png' width=700 height=300>

## Adversary Variational AutoEncoder

<img src='imgs/layers_AAE.png' width=700 height=300>

## Prerequisites
- Linux or macOS
- python3+
- Pytorch 0.4+

## Installation
```bash
git clone https://github.com/CY-Jeong/anomaly-detection-mvtec.git
cd anomaly-detection-mvtec
```
- Pip users, ```pip install -r requirements.txt```
- Conda users, ```conda env create -f environment.yml```

## Usage
First, download MVTEC datasets.
```bash
mkdir [data location]
cd [data location]
wget ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
tar Jxvf mvtec_anomaly_detection.tar.xz
```
To train a model
```bash
python train.py --data_dir [data location] --save_dir [save location] --model [aae | cae]
```
If you choose CAE, the n_epochs should be 1000 more, or 60 epochs in aae.
To test a model
```bash
python test.py --data_dir [data location] --save_dir [save location] --model [aae | cae]
```

# Results

<img src='imgs/result1.png' width=900 height=300>
<img src='imgs/result2.png' width=900 height=300>
<img src='imgs/result3.png' width=900 height=300>
<img src='imgs/result4.png' width=900 height=300>
<img src='imgs/result5.png' width=900 height=300>
<img src='imgs/result6.png' width=900 height=300>

# Related Projects
[Anogan](https://arxiv.org/pdf/1703.05921.pdf)|[VAE-Anomaly-Detection](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
[Deep Anomaly Detection Using Geometric Transformations](https://arxiv.org/pdf/1805.10917.pdf)<br>
[Out-of-Distribution NN](https://arxiv.org/pdf/1610.02136.pdf)|[Out-of-Distribution GAN](https://arxiv.org/pdf/1909.11480.pdf)


