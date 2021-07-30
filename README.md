# Image Classification Pipeline

* [Introduction](#introduction)
* [Features](#features)
* [Model Zoo](#model-zoo)
* [Configuration](#configuration)
* [Training](#training)
    * [Single GPU](#single-gpu)
    * [Multiple GPUs](#multiple-gpus)
    * [Knowledge Distillation](#knowledge-distillation)
* [Evaluation](#evaluation)
* [Inference](#inference)

## Introduction

There are a lot of great repositories for image classification models but most of them are quite complicated if you want to modify or only need necessary parts. 

In this repository, a complete training, evaluation and inference pipeline for image classfication is written for the purpose of easy to understand and modify. 

If you want to use a custom model, custom dataset and other training configurations like optimizers, schedulers, etc., you can modify easily after taking a quick look at the codes.

## Features

Datasets
* [ImageNet](https://image-net.org/)

Models
* CNN
    * [HRNet](https://arxiv.org/abs/1908.07919)
    * [MobileNetV3](https://arxiv.org/abs/1905.02244v5) 
    * [MobileNetV2](https://arxiv.org/abs/1801.04381v4)
    * [ResNet](https://arxiv.org/abs/1512.03385)
    * [EfficientNetV2](https://arxiv.org/abs/2104.00298v3)

* Transformer
    * [CSWin](https://arxiv.org/abs/2107.00652v2) (Coming Soon)
    * [VOLO](https://arxiv.org/abs/2106.13112v1) (Coming Soon)
    * [Refiner](https://arxiv.org/abs/2106.03714v1) (Coming Soon)
    * [XCiT](https://arxiv.org/abs/2106.09681)
    * [CaiT](https://arxiv.org/abs/2103.17239) 
    * [LV-ViT](https://arxiv.org/abs/2104.10858v2)
    * [DeiT](https://arxiv.org/abs/2012.12877) 
    * [ViT](https://arxiv.org/abs/2010.11929v2)
    
* MLP
    * [CycleMLP](https://arxiv.org/abs/2107.10224) 
    * [ViP](https://arxiv.org/abs/2106.12368v1)
    * [gMLP](https://arxiv.org/abs/2105.08050v2) 
    * [ResMLP](https://arxiv.org/abs/2105.03404) 
    * [MLP Mixer](https://arxiv.org/abs/2105.01601)

Knowledge Distillation
* [Vanilla KD](https://arxiv.org/abs/1503.02531)

## Model Zoo

Model | ImageNet1k Top-1 Acc (%) | Params (M)  | GFLOPs | Throughput (image/s) | Peak Mem (MB) | Weights
--- | --- | --- | --- | --- | --- | --- 
MobileNetV2 | 72.0 | **3.4** | - | - | - | N/A
MobileNetV3 | 75.2 | **5.4** | - | - | - | N/A
EffNetV2-S | 83.9 | **24** | - | - | - | N/A
EffNetV2-M | **85.1** | 55 | - | - | - | N/A
EffNetV2-L | **85.7** | 121 | - | - | - | N/A
ViT-B (384) | 77.9 | 86 | 55.4 | 85.9 | - | N/A
DeiT-T | 74.5 | **6** | - | - | - | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
DeiT-S | 81.2 | **22** | **4.6** | **940.4** | **217.2** | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
DeiT-B | 83.4 | 87 | **17.5** | 292.3 | 573.7 | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
LV-ViT-S | 83.3 | **26** | 22.2 | - | - | [download](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar)
LV-ViT-M | **84.0** | 56 | 42.2 | - | - | [download](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar)
CaiT-S24 (384) | **85.1** | 47 | 32.2 | - | - | [download](https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing)
CaiT-S36 (384) | **85.4** | 68 | 48 | - | - | [download](https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing)
Refiner-S | 83.6 | **25** | - | - | - | N/A
Refiner-M | **84.6** | 55 | - | - | - | N/A
Refiner-L | **84.9** | 81 | - | - | - | N/A
XCiT-T24 | 82.6 | **12** | - | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-S24 | **84.9** | **26** | - | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-M24 | **85.1** | 84 | - | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-L24 | **85.4** | 189 | - | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
CSWin-T | 82.7 | **23** | **4.3** | - | - | N/A
CSWin-S | 83.6 | 35 | **6.9** | - | - | N/A
CSWin-B | **84.2** | 78 | **15.0** | - | - | N/A
VOLO-D1 | **84.2** | **27** | **6.8** | - | -  | N/A
VOLO-D2 | **85.2** | 59 | **14.1** | - | -  | N/A
VOLO-D3 | **85.4** | 86 | 20.6 | - | - | N/A
VOLO-D4 | **85.7** | 193 | 43.8 | - | -  | N/A
VOLO-D5 | **86.1** | 296 | - | - | - | N/A
Mixer-B | 76.4 | 59 | **12.7** | - | - | N/A
ResMLP-S12 | 76.6 | **15** | **3.0** | **1415.1** | **179.5** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth)
ResMLP-S24 | 79.4 | 30 | **6.0** | **715.4** | **235.3** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth)
ResMLP-S36 | 81.0 | 116 | 23.0 | 231.3 | 663.0 | [download](https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth) 
gMLP-S | 79.6 | **20** | **4.5** | - | - | N/A
gMLP-B | 81.6 | 73 | **15.8** | - | - | N/A
ViP-S | 81.5 | **25** | **6.9** | **719** | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-M | 82.7 | 55 | **16.3** | 418 | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-L | 83.2 | 88 | 24.4 | 298 | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
CycleMLP-B1 | 78.9 | 15 | **2.1** | - | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth)
CycleMLP-B2 | 81.6 | 27 | **3.9** | - | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B2.pth)
CycleMLP-B3 | 82.4 | 38 | **6.9** | - | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B3.pth)
CycleMLP-B4 | 83.0 | 52 | **10.1** | - | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B4.pth)
CycleMLP-B5 | 83.2 | 76 | **12.3** | - | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B5.pth)

> Notes: All models' results are from their papers or official repos. Throughput is measured with V100GPU. Weights are converted from official repos. Only models trained on ImageNet1k are compared. Huge parameters models (>200M) are not included. If the distilled version of the model exists, its result is reported. Image size is 224x224 unless otherwise specified.


## Configuration 

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

## Training

### Single GPU
```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### Multiple GPUs

Traing with 2 GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### Knowledge Distillation

Change `ENABLE` field in `KD` of the configuration file to `True` and also change the additional parameters. The weights file for the teacher model must be supplied via `PRETRAINED` field.

The training command is the same as in above.

## Evaluation

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Inference

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Others

* For optimizing pytorch models for deployment, see [torch_optimize](https://github.com/sithu31296/torch_optimize).