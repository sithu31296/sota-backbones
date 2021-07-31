# <div align="center">Image Classification</div>
<p align="center"> A collection of SOTA Image Classification Models implemented in PyTorch.  </p>

## <div align="center">Model Zoo</div>

#### ImageNet-1k Comparison

Model | Top-1 Accuracy <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | Throughput <br><sup>(image/s) | Weights
--- | --- | --- | --- | --- | --- 
[MobileNetV2](https://arxiv.org/abs/1905.02244v5) | 72.0 | **3.4** | - | - | [download](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)
[MobileNetV3](https://arxiv.org/abs/1801.04381v4) | 75.2 | **5.4** | - | - | [download](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth)
[EffNetV2-S](https://arxiv.org/abs/2104.00298v3) | 83.9 | 24 | - | - | N/A
EffNetV2-M | **85.1** | 55 | - | - | N/A
EffNetV2-L | **85.7** | 121 | - | - | N/A
 | | | | |
[ViT-B](https://arxiv.org/abs/2010.11929v2) (384) | 77.9 | 86 | 55.4 | 85.9 | N/A
[DeiT-T](https://arxiv.org/abs/2012.12877) | 74.5 | **6** | - | - | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
DeiT-S | 81.2 | 22 | **4.6** | **940.4** | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
DeiT-B | 83.4 | 87 | 17.5 | 292.3 | [download](https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing)
[LV-ViT-S](https://arxiv.org/abs/2104.10858v2) | 83.3 | 26 | 22.2 | - | [download](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar)
LV-ViT-M | 84.0 | 56 | 42.2 | - | [download](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar)
[CaiT-S24](https://arxiv.org/abs/2103.17239) (384) | **85.1** | 47 | 32.2 | - | [download](https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing)
CaiT-S36 (384) | **85.4** | 68 | 48 | - | [download](https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing)
[XCiT-T24](https://arxiv.org/abs/2106.09681) | 82.6 | **12** | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-S24 | 84.9 | 26 | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-M24 | **85.1** | 84 | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
XCiT-L24 | **85.4** | 189 | - | - | [download](https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing)
[CSWin-T](https://arxiv.org/abs/2107.00652v2) | 82.7 | 23 | **4.3** | - | [download](https://drive.google.com/drive/folders/1kVTZwgJ0uCTynUa2vOJ5SUgL2R7PyNLa?usp=sharing)
CSWin-S | 83.6 | 35 | **6.9** | - | [download](https://drive.google.com/drive/folders/1kVTZwgJ0uCTynUa2vOJ5SUgL2R7PyNLa?usp=sharing)
CSWin-B | 84.2 | 78 | 15.0 | - | [download](https://drive.google.com/drive/folders/1kVTZwgJ0uCTynUa2vOJ5SUgL2R7PyNLa?usp=sharing)
[VOLO-D1](https://arxiv.org/abs/2106.13112v1) | 84.2 | 27 | **6.8** | - | [download](https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar)
VOLO-D2 | **85.2** | 59 | 14.1 | - | [download](https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar)
VOLO-D3 | **85.4** | 86 | 20.6 | - | [download](https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar)
VOLO-D4 | **85.7** | 193 | 43.8 | - | [download](https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar)
 | | | | |
[Mixer-B](https://arxiv.org/abs/2105.01601) | 76.4 | 59 | 12.7 | - | N/A
[ResMLP-S12](https://arxiv.org/abs/2105.03404) | 76.6 | **15** | **3.0** | **1415.1** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth)
ResMLP-S24 | 79.4 | 30 | **6.0** | **715.4** | [download](https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth)
ResMLP-S36 | 81.0 | 116 | 23.0 | 231.3 | [download](https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth) 
[gMLP-S](https://arxiv.org/abs/2105.08050v2) | 79.6 | 20 | **4.5** | - | N/A
gMLP-B | 81.6 | 73 | 15.8 | - | N/A
[ViP-S](https://arxiv.org/abs/2106.12368v1) | 81.5 | 25 | **6.9** | **719** | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-M | 82.7 | 55 | 16.3 | 418 | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
ViP-L | **83.2** | 88 | 24.4 | 298 | [download](https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing)
[CycleMLP-B1](https://arxiv.org/abs/2107.10224) | 78.9 | **15** | **2.1** | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth)
CycleMLP-B2 | 81.6 | 27 | **3.9** | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B2.pth)
CycleMLP-B3 | 82.4 | 38 | **6.9** | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B3.pth)
CycleMLP-B4 | **83.0** | 52 | 10.1 | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B4.pth)
CycleMLP-B5 | **83.2** | 76 | 12.3 | - | [download](https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B5.pth)

<details>
  <summary>Table Notes <small>(click to expand)</small></summary>

* Table contains 3 sections; CNN, Transformer and MLP based models.
* Models' results are from their papers or official repos. 
* Throughput is measured with V100GPU. 
* Weights are converted from official repos. 
* Only models trained on ImageNet1k are compared. 
* Huge parameters models (>200M) are not included. 
* If the distilled version of the model exists, its result is reported. 
* Image size is 224x224, unless otherwise specified.
</details>

<details>
  <summary>Model Summary <small>(click to expand)</small></summary>


</details>

## <div align="center">Usage</div>

<details>
  <summary>Configuration <small>(click to expand)</small></summary>

Create a configuration file in `configs`. Sample configuration for ImageNet dataset can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<details>
  <summary>Training <small>(click to expand)</small></summary>

Train with 1 GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

Train with 2 GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Training with KD <small>(click to expand)</small></summary>

Change `ENABLE` field in `KD` of the configuration file to `True` and also change the additional parameters. The weights file for the teacher model must be supplied via `PRETRAINED` field.

The training command is the same as in above.

</details>


<details>
  <summary>Evaluation <small>(click to expand)</small></summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>


<details>
  <summary>Inference <small>(click to expand)</small></summary>

Make sure to set `MODEL_PATH` of the configuration file to model's weights.

```bash
$ python tools/infer.py --cfg configs/CONFIG_FILE_NAME.yaml
```

</details>

<details>
  <summary>Optimization <small>(click to expand)</small></summary>

For optimizing these models for deployment, see [torch_optimize](https://github.com/sithu31296/torch_optimize).

</details>

<details>
  <summary>References <small>(click to expand)</small></summary>



</details>