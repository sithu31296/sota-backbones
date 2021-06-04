# FastCls

* [Introduction](##Introduction)
* [Features](##Features)
* [Model Comparison](##Model-Comparison)
* [Configuration](##Configuration)
* [Training](##Training)
* [Evaluation](##Evaluation)
* [Inference](##Inference)

## Introduction

There are a lot of great repositories for semantic segmentation models but most of them are quite complicated if you want to modify or only need necessary parts. 

In this repository, a complete training, evalaution and inference pipeline for image classfication is written for the purpose of easy to understand and modify. 

If you want to use a custom model, custom dataset and other training configurations like optimizers, schedulers, etc., you can modify easily after taking a quick look at the codes.

## Features

Datasets
* [ImageNet](https://image-net.org/)

Models
* CNN
    * [ResNet](https://arxiv.org/abs/1512.03385)
    * [HRNet](https://arxiv.org/abs/1908.07919) 
* Transformer
    * [ViT](https://arxiv.org/pdf/2010.11929v2.pdf) (Coming Soon)
    * [DeiT](https://arxiv.org/abs/2012.12877) (Coming Soon)
    * [Swin](https://arxiv.org/abs/2103.14030) (Coming Soon)
    * [LV-ViT](https://arxiv.org/abs/2104.10858v2) (Coming Soon)
    * [CaiT](https://arxiv.org/abs/2103.17239) (Coming Soon)
    
* MLP
    * [MLP Mixer](https://arxiv.org/abs/2105.01601)
    * [ResMLP](https://arxiv.org/abs/2105.03404) (Coming Soon)
    * [gMLP](https://arxiv.org/abs/2105.08050v2) (Coming Soon)


Features coming soon:
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)


## Model Comparison

Model | ImageNet1k Top-1 Accuracy (%) | Params (M)  | FLOPs (B) | Train size | Test size
--- | --- | --- | --- | --- | ---
ViT-B/16 | 77.9 | 86 | 55.4 | 224 | 384
ViT-L/16 | 76.5 | 307 | 190.7 | 224 | 384
DeiT-S | 79.9 | 22 | 4.6 | 224 | 224
DeiT-B | 83.1 | 86 | 55.4 | 224 | 384
Swin-S | 83.0 | 50 | 8.7 | 224 | 224
Swin-B | 84.2 | 88 | 47.0 | 224 | 384
LV-ViT-S | 84.4 | 26 | 22.2 | 224 | 384
LV-ViT-M | 85.4 | 56 | 42.2 | 224 | 384
CaiT-S-36 | 85.4 | 68 | 48 | 224 | 384
CaiT-M-36 | 86.3 | 271 | 173.3 | 224 | 384
Mixer-B/16 | 76.4 | 59 | 12.7 | 224 | 224
Mixer-L/16 | 71.8 | 207 | 44.8 | 224 | 224
ResMLP-12 | 76.6 | 15 | 3.0 | 224 | 224
ResMLP-24 | 79.4 | 30 | 6.0 | 224 | 224
ResMLP-36 | 79.7 | 45 | 8.9 | 224 | 224
gMLP-S | 79.6 | 20 | 4.5 | 224 | 224
gMLP-B | 81.6 | 73 | 15.8 | 224 | 224


## Configuration 

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

## Training

```bash
$ python train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

```bash
$ python val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Inference

```bash
$ python infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

