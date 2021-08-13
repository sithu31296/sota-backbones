# <div align="center">SOTA Image Classification Models in PyTorch</div>

![visiontransformer](assests/vit_banner.png)

## <div align="center">Model Zoo</div>

[efficientv2]: https://arxiv.org/abs/2104.00298
[cait]: https://arxiv.org/abs/2103.17239
[xcit]: https://arxiv.org/abs/2106.09681
[cswin]: https://arxiv.org/abs/2107.00652v2
[volo]: https://arxiv.org/abs/2106.13112v1
[gfnet]: https://arxiv.org/abs/2107.00645
[pvtv2]: https://arxiv.org/abs/2106.13797
[crossformer]: https://arxiv.org/abs/2108.00154
[lstransformer]: https://arxiv.org/abs/2107.02192
[shuffle]: https://arxiv.org/abs/2106.03650
[conformer]: https://arxiv.org/abs/2105.03889v1
[rest]: https://arxiv.org/abs/2105.13677v3
[nest]: https://arxiv.org/abs/2105.12723v2

[vip]: https://arxiv.org/abs/2106.12368v1
[cyclemlp]: https://arxiv.org/abs/2107.10224

[caitw]: https://drive.google.com/drive/folders/1YrbN3zdz1jykz5D-CY6ND7A7schH8E19?usp=sharing
[xcitw]: https://drive.google.com/drive/folders/10lvfB8sXdRuZve5xn6pebJN6TT2GaYhP?usp=sharing
[cswinw]: https://drive.google.com/drive/folders/1kVTZwgJ0uCTynUa2vOJ5SUgL2R7PyNLa?usp=sharing
[volod1]: https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar
[volod2]: https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar
[volod3]: https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar
[volod4]: https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar
[restm]: https://drive.google.com/drive/folders/1H6QUZsKYbU6LECtxzGHKqEeGbx1E8uQ9
[gfnett]: https://drive.google.com/file/d/1Nrq5sfHD9RklCMl6WkcVrAWI5vSVzwSm/view?usp=sharing
[gfnets]: https://drive.google.com/file/d/1w4d7o1LTBjmSkb5NKzgXBBiwdBOlwiie/view?usp=sharing
[gfnetb]: https://drive.google.com/file/d/1F900_-yPH7GFYfTt60xn4tu5a926DYL0/view?usp=sharing
[pvt1]: https://drive.google.com/file/d/1aM0KFE3f-qIpP3xfhihlULF0-NNuk1m7/view?usp=sharing
[pvt2]: https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing
[pvt3]: https://drive.google.com/file/d/1PzTobv3pu5R3nb3V3lF6_DVnRDBtSmmS/view?usp=sharing
[pvt4]: https://drive.google.com/file/d/1LW-0CFHulqeIxV2cai45t-FyLNKGc5l0/view?usp=sharing
[pvt5]: https://drive.google.com/file/d/1TKQIdpOFoFs9H6aApUNJKDUK95l_gWy0/view?usp=sharing
[crosst]: https://drive.google.com/file/d/1YSkU9enn-ITyrbxLH13zNcBYvWSEidfq/view?usp=sharing
[crosss]: https://drive.google.com/file/d/1RAkigsgr33va0RZ85S2Shs2BhXYcS6U8/view?usp=sharing
[crossb]: https://drive.google.com/file/d/1bK8biVCi17nz_nkt7rBfio_kywUpllSU/view?usp=sharing
[shufflet]: https://drive.google.com/drive/folders/1goDJtcnxgBAcHhZnNwrgOlG_WBftpmOS?usp=sharing
[shuffles]: https://drive.google.com/drive/folders/1GUBBQyDldY145vDiK-BHqivmpj3K6HK2?usp=sharing
[shuffleb]: https://drive.google.com/drive/folders/1x0biaJRdN4nxLmp_3lQcA_6hO_sDBoUM?usp=sharing
[vipw]: https://drive.google.com/drive/folders/1l2XWrzqeP5n3tIm4O1jkd727j_mVoOf1?usp=sharing
[cycleb1]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B1.pth
[cycleb2]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B2.pth
[cycleb3]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B3.pth
[cycleb4]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B4.pth
[cycleb5]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B5.pth
[conformert]: https://drive.google.com/file/d/19SxGhKcWOR5oQSxNUWUM2MGYiaWMrF1z/view?usp=sharing
[conformers]: https://drive.google.com/file/d/1mpOlbLaVxOfEwV4-ha78j_1Ebqzj2B83/view?usp=sharing
[conformerb]: https://drive.google.com/file/d/1oeQ9LSOGKEUaYGu7WTlUGl3KDsQIi0MA/view?usp=sharing

Model | ImageNet-1k Top-1 Acc <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | <sup> Variants & Weights
--- | --- | --- | --- | --- 
[EfficientNetv2][efficientv2] | 83.9`\|`85.1`\|`85.7 | 22`\|`54`\|`120 | 9`\|`24`\|`53 | S\|M\|L
||
[CaiT][cait]* (384) | 85.1`\|`85.4 | 47`\|`68 | 32`\|`48 | [S24][caitw]\|[S36][caitw]
[XCiT][xcit]* | 82.6`\|`84.9`\|`85.1`\|`85.4 | 12`\|`48`\|`84`\|`189 | 9`\|`36`\|`64`\|`142 | [T24][xcitw]\|[S24][xcitw]\|[M24][xcitw]\|[L24][xcitw]
[VOLO][volo]^ | 84.2`\|`85.2`\|`85.4`\|`85.7 | 27`\|`59`\|`86`\|`193 | 7`\|`14`\|`21`\|`44 | [D1][volod1]\|[D2][volod2]\|[D3][volod3]\|[D4][volod4]
||
[GFNet][gfnet] | 80.1`\|`81.5`\|`82.9 | 15`\|`32`\|`54 | 2`\|`5`\|`8 | [H-T][gfnett]\|[H-S][gfnets]\|[H-B][gfnetb]
[ResT][rest] | 79.6`\|`81.6`\|`83.6 | 14`\|`30`\|`52 | 2`\|`4`\|`8 | [S][restm]\|[B][restm]\|[L][restm]
[NesT][nest] | 81.5`\|`83.3`\|`83.8 | 17`\|`38`\|`68 | 6`\|`10`\|`18 | T\|S\|B
[PVTv2][pvtv2] | 78.7`\|`82.0`\|`83.1`\|`83.6`\|`83.8 | 14`\|`25`\|`45`\|`63`\|`82 | 2`\|`4`\|`7`\|`10`\|`12 | [B1][pvt1]\|[B2][pvt2]\|[B3][pvt3]\|[B4][pvt4]\|[B5][pvt5]
||
[CrossFormer][crossformer] | 81.5`\|`82.5`\|`83.4`\|`84.0 | 28`\|`31`\|`52`\|`92 | 3`\|`5`\|`9`\|`16 | [T][crosst]\|[S][crosss]\|[B][crossb]\|L
[Shuffle][shuffle] | 82.4`\|`83.6`\|`84.0 | 28`\|`50`\|`88 | 5`\|`9`\|`16 | [T][shufflet]\|[S][shuffles]\|[B][shuffleb]
[Conformer][conformer] | 81.3`\|`83.4`\|`84.1 | 24`\|`38`\|`83 | 5`\|`11`\|`23 | [T][conformert]\|[S][conformers]\|[B][conformerb]
[CSWin][cswin] | 82.7`\|`83.6`\|`84.2 | 23`\|`35`\|`78 | 4`\|`7`\|`15 | [T][cswinw]\|[S][cswinw]\|[B][cswinw]
[LongShort][lstransformer] | 83.8`\|`84.1 | 40`\|`56 | 9`\|`13 | M\|B
||
[ViP][vip] | 81.5`\|`82.7`\|`83.2 | 25`\|`55`\|`88 | 7`\|`16`\|`24 | [S][vipw]\|[M][vipw]\|[L][vipw]
[CycleMLP][cyclemlp] | 78.9`\|`81.6`\|`82.4`\|`83.0`\|`83.2 | 15`\|`27`\|`38`\|`52`\|`76 | 2`\|`4`\|`7`\|`10`\|`12 | [B1][cycleb1]\|[B2][cycleb2]\|[B3][cycleb3]\|[B4][cycleb4]\|[B5][cycleb5]

<details>
  <summary>Table Notes <small>(click to expand)</small></summary>

* Image size is 224x224, unless otherwise specified. EfficientNetv2 uses progressive learning (image size from 128 to 380).
* Only models trained on ImageNet1k are compared. 
* (Parameters > 200M) Models are not included. 
* `*` model uses knowledge distillation whereas `^` model uses token labeling.

</details>


## <div align="center">Usage</div>

<details>
  <summary>Dataset Preparation <small>(click to expand)</small></summary>

For standard ImageNet dataset, the dataset structure should look like this:

```
imagenet
|__ train
    |__ class1
        |__ img1.jpg
        |__ ...
    |__ class2
        |__ img10.jpg
        |__ ...
|__ val
    |__ class1
        |__ img4.jpg
        |__ ...
    |__ class2
        |__ img12.jpg
        |__ ...
```

You can also use this structure for custom datasets.


</details>

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