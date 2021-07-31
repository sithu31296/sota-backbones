# <div align="center">Image Classification</div>
<p align="center"> A collection of SOTA Image Classification Models implemented in PyTorch.  </p>

## <div align="center">Model Zoo</div>

#### ImageNet-1k Comparison

[mobilenetv2]: https://arxiv.org/abs/1905.02244v5
[mobilenetv3]: https://arxiv.org/abs/1801.04381v4
[efficientv2]: https://arxiv.org/abs/2104.00298v3

[vit]: https://arxiv.org/abs/2010.11929v2
[deit]: https://arxiv.org/abs/2012.12877
[lvvit]: https://arxiv.org/abs/2104.10858v2
[cait]: https://arxiv.org/abs/2103.17239
[xcit]: https://arxiv.org/abs/2106.09681
[cswin]: https://arxiv.org/abs/2107.00652v2
[volo]: https://arxiv.org/abs/2106.13112v1

[mixer]: https://arxiv.org/abs/2105.01601
[resmlp]: https://arxiv.org/abs/2105.03404
[gmlp]: https://arxiv.org/abs/2105.08050v2
[vip]: https://arxiv.org/abs/2106.12368v1
[cyclemlp]: https://arxiv.org/abs/2107.10224

[mobilenetv2w]: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
[mobilenetv3w]: https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
[deitw]: https://drive.google.com/drive/folders/1nhj-RSAHcpN3e5G0eryKBcnUwlyE_YYv?usp=sharing
[lvvits]: https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar
[lvvitm]: https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar
[caitw]: https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing
[xcitw]: https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing
[cswinw]: https://drive.google.com/drive/folders/1kVTZwgJ0uCTynUa2vOJ5SUgL2R7PyNLa?usp=sharing
[volod1]: https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar
[volod2]: https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar
[volod3]: https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar
[volod4]: https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar
[resmlps12]: https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth
[resmlps24]: https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth
[resmlps36]: https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth
[vipw]: https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing
[cycleb1]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth
[cycleb2]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B2.pth
[cycleb3]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B3.pth
[cycleb4]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B4.pth
[cycleb5]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B5.pth

Model | Top-1 Accuracy <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | <sup> Variants & Weights
--- | --- | --- | --- | --- 
[Mobilenetv2][mobilenetv2] | 72.0 | 3.4 | - | [v2][mobilenetv2w]
[Mobilenetv3][mobilenetv3] | 75.2 | 5.4 | - | [v3][mobilenetv3w]
[Effnetv2][efficientv2] | 83.9`\|`85.1`\|`85.7 | 24`\|`55`\|`121 | - | S \| M \| L
 | | | | |
[ViT][vit] (384) | 77.9 | 86 | 55 | B
[DeiT][deit] | 74.5`\|`81.2`\|`83.4 | 6`\|`22`\|`87 | -`\|`5`\|`18 | [T\|S\|B][deitw]
[LV-ViT][lvvit] | 83.3`\|`84.0 | 26`\|`56 | 22`\|`42 | [S][lvvits] \| [M][lvvitm]
[CaiT][cait] (384) | 85.1`\|`85.4 | 47`\|`68 | 32`\|`48 | [S24\|S36][caitw]
[XCiT][xcit] | 82.6`\|`84.9`\|`85.1`\|`85.4 | 12`\|`26`\|`84`\|`189 | - | [T\|S\|M\|L][xcitw]
[CSWin][cswin] | 82.7`\|`83.6`\|`84.2 | 23`\|`35`\|`78 | 4`\|`7`\|`15 | [T\|S\|B][cswinw]
[VOLO][volo] | 84.2`\|`85.2`\|`85.4`\|`85.7 | 27`\|`59`\|`86`\|`193 | 7`\|`14`\|`21`\|`44 | [D1][volod1] \| [D2][volod2] \| [D3][volod3] \| [D4][volod4]
 | | | | |
[Mixer][mixer] | 76.4 | 59 | 13 | B
[ResMLP][resmlp] | 76.6`\|`79.4`\|`81.0 | 15`\|`30`\|`116 | 3`\|`6`\|`23 | [S12][resmlps12] \| [S24][resmlps24] \| [S36][resmlps36]
[gMLP][gmlp] | 79.6`\|`81.6 | 20`\|`73 | 5`\|`16 | S \| B
[ViP][vip] | 81.5`\|`82.7`\|`83.2 | 25`\|`55`\|`88 | 7`\|`16`\|`24 | [S\|M\|L][vipw]
[CycleMLP][cyclemlp] | 78.9`\|`81.6`\|`82.4`\|`83.0`\|`83.2 | 15`\|`27`\|`38`\|`52`\|`76 | 2`\|`4`\|`7`\|`10`\|`12 | [B1][cycleb1] \| [B2][cycleb2] \| [B3][cycleb3] \| [B4][cycleb4] \| [B5][cycleb5]

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

* CNN models' accuracy = 83~86
* Transformer models' accuracy = 83~85
* MLP models' accuracy = 81~83
* Some models use additional token labelling during training. (LV-ViT, VOLO)
* Some models use knowledge distillation to improve their accuracy. (CaiT, XCiT)

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