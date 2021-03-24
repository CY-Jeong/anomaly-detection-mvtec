# Anomaly-Detection
Pytorch implemetation of **Anomaly Detection** which detect not normal images in industrial datasets [mvtec](https://www.mvtec.com/)<br>
It has only simple layers but if you want to look out mvtec paper click [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf).<br>
Actually I didn't work well with mvtecAD as I think finding automatic threshold in finetune stage.
So this code is implemeted only manualy fix the threshold. That's weekness.

## Tensorflow version
Tensorflow implementation of **[MVTEC-AD](https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection)** which is implemetation in mvtec paper linked above.


## Simple CNN models(AutoEncoder)

<img src='imgs/layers_AE.png' align="left" width=1000>

## Adversary Variational AutoEncoder

<img src='imgs/layers_AAE.png' align="right" width=1000>

## Prerequisites
- python3+
- Pytorch 1.4+
- environments.yml

## Usage
First, download MVTEC datasets.
```bash
mkdir Downloads
cd Downloads
wget ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
tar Jxvf mvtec_anomaly_detection.tar.xz
```
To train a model
```bash
python train.py
```

To test a model
```bash
python test.py
```

# Results

<img src='imgs/layers_AE.png' width=384>

# Related Projects
[Anogan](https://arxiv.org/pdf/1703.05921.pdf)|[VAE-Anomaly-Detection](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
[Deep Anomaly Detection Using Geometric Transformations](https://arxiv.org/pdf/1805.10917.pdf)<br>
[Out-of-Distribution NN](https://arxiv.org/pdf/1610.02136.pdf)|[Out-of-Distribution GAN](https://arxiv.org/pdf/1909.11480.pdf)


