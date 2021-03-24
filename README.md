# Anomaly-Detection
Pytorch implemetation of **Anomaly Detection** which detect not normal images in industrial datasets [mvtec](https://www.mvtec.com/)<br>
It has only simple layers but if you want to look out mvtec paper click [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)

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

<img src='imgs/layers_AE.png' align="left" width=384>

