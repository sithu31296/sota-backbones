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
    * [PyTorch Inference](#pytorch-inference)
    * [ONNX Inference](#onnx-inference)
    * [TFLite Inference](#tflite-inference)
* [Optimization](#optimization)
    * [Quantization](#quantization)
    * [Pruning](#pruning)
* [Other Pipelines](#other-pipelines)

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
    * [MobileNetV3](https://arxiv.org/abs/1905.02244v5) (Coming Soon)
    * [MobileNetV2](https://arxiv.org/abs/1801.04381v4)
    * [ResNet](https://arxiv.org/abs/1512.03385)
    * [EfficientNetV2](https://arxiv.org/abs/2104.00298v3) (Coming Soon)

* Transformer
    * [CSWin](https://arxiv.org/abs/2107.00652v2) (Coming Soon)
    * [VOLO](https://arxiv.org/abs/2106.13112v1) (Coming Soon)
    * [Refiner](https://arxiv.org/abs/2106.03714v1) (Coming Soon)
    * [XCiT](https://arxiv.org/abs/2106.09681)
    * [CaiT](https://arxiv.org/abs/2103.17239) 
    * [CvT](https://arxiv.org/abs/2103.15808) (Coming Soon)
    * [LV-ViT](https://arxiv.org/abs/2104.10858v2)
    * [DeiT](https://arxiv.org/abs/2012.12877) 
    * [ViT](https://arxiv.org/abs/2010.11929v2)
    
* MLP
    * [ASMLP](https://arxiv.org/abs/2107.08391v1) (Coming Soon)
    * [ViP](https://arxiv.org/abs/2106.12368v1)
    * [gMLP](https://arxiv.org/abs/2105.08050v2) 
    * [ResMLP](https://arxiv.org/abs/2105.03404) 
    * [MLP Mixer](https://arxiv.org/abs/2105.01601)

Knowledge Distillation
* [Vanilla KD](https://arxiv.org/abs/1503.02531)
* [TAKD](https://arxiv.org/abs/1902.03393) (Coming Soon)
* [CRD](http://arxiv.org/abs/1910.10699) (Coming Soon)

Training
* [AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
* [DDP](https://pytorch.org/docs/stable/notes/ddp.html) 

Model Conversion
* [ONNX](https://github.com/onnx/onnx)
* [TFLite](https://www.tensorflow.org/lite)
* [TensorRT](https://github.com/NVIDIA/TensorRT) (Coming Soon)

Model Inspection
* [Benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html) [[Implementation](./tools/inspect/benchmark.py)]
* [Profiler](https://pytorch.org/docs/stable/profiler.html) [[Implementation](./tools/inspect/model_profile.py)]

Optimization
* [Quantization](https://pytorch.org/docs/stable/quantization.html) 
* [Mobile Optimizer](https://pytorch.org/docs/stable/mobile_optimizer.html) 
* [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

Deployment
* [Flask+Vue App]() (Coming Soon)
* [FastAPI App]() (Coming Soon)


## Model Zoo

Model | ImageNet1k Top-1 Accuracy (%) | Params (M)  | FLOPs (B) | Throughput (image/s) | Peak Memory (MB) | Weights
--- | --- | --- | --- | --- | --- | --- 
ResNet-50 | 77.15 | - | - | - | - | N/A
ResNet-101 | 78.25 | - | - | - | - | N/A
ResNet-152 | 78.57 | 60 | - | - | - | N/A
MobileNetV2 | 72.0 | **3.4** | - | - | - | N/A
MobileNetV3-S | 67.4 | **2.5** | - | - | - | N/A
MobileNetV3-L | 75.2 | **5.4** | - | - | - | N/A
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
CvT-13 | 81.6 | **20** | **4.5** | - | - | N/A
CvT-21 | 82.5 | 32 | **7.1** | - | - | N/A
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
Mixer-B | 76.4 | 59 | **12.7** | - | - | N/A
ResMLP-S12 | 76.6 | **15** | **3.0** | **1415.1** | **179.5** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth)
ResMLP-S24 | 79.4 | 30 | **6.0** | **715.4** | **235.3** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth)
ResMLP-S36 | 81.0 | 116 | 23.0 | 231.3 | 663.0 | [download](https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth) 
gMLP-S | 79.6 | **20** | **4.5** | - | - | N/A
gMLP-B | 81.6 | 73 | **15.8** | - | - | N/A
ViP-S | 81.5 | **25** | - | **719** | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-M | 82.7 | 55 | - | 418 | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-L | 83.2 | 88 | - | 298 | - | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ASMLP-T | 81.3 | **28** | **4.4** | **1047** | - | N/A
ASMLP-S | 83.1 | 50 | **8.5** | 619 | - | N/A
ASMLP-B | 83.3 | 88 | **15.2** | 455 | - | N/A

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

### PyTorch Inference

```bash
$ python tools/inference/pt_infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### ONNX Inference

Install necessary tools:
* ONNX-Simplifier via `pip install onnx-simplifier`.
* ONNXRuntime via `pip install onnxruntime`.

Convert the PyTorch model to onnx model:

```bash
$ python tools/convert/pt_to_onnx.py --cfg configs/CONFIG_FILE_NAME.yaml
```

Run an inference with:

```bash
$ python tools/inference/onnx_infer.py --model-path ONNX_MODEL_PATH.onnx --file TEST_IMG_DIR
```

### TFLite Inference

Install necessary tools:
* ONNX-Simplifier via `pip install onnx-simplifier`.
* Tensorflow2 via `pip install tensorflow`.
* ONNX-Tensorflow via `pip install onnx-tf`.

Convert the PyTorch model to tflite model:

```bash
$ python tools/convert/pt_to_tflite.py --cfg configs/CONFIG_FILE_NAME.yaml
```

Run an inference with:

```bash
$ python tools/inference/tflite_infer.py --model-path TFLITE_MODEL_PATH.tflite --file TEST_IMG_DIR
```

## Optimization

### Quantization

Change `QUANTIZE` parameters in the configuration file and run the following. The quantized model will be saved in `SAVE_DIR`.

```bash
$ python tools/optimize/quantize.py --cfg configs/CONFIG_FILE_NAME.yaml
```

### Pruning

Change `PRUNE` parameters in the configuration file and run the following. The pruned model will be saved in `SAVE_DIR`.

```bash
$ python tools/optimize/pruning.py --cfg configs/CONFIG_FILE_NAME.yaml
```


## Other Pipelines

* [Semantic-Segmentation-Pipeline](https://github.com/sithu31296/Semantic-Segmentation-Pipeline)
* [Re-Identification-Pipeline](https://github.com/sithu31296/Re-Identification-Pipeline)