# <div align="center">SOTA Image Classification Models in PyTorch</div>

<div align="center">
<p>Intended for easy to use and integrate SOTA image classification models into object detection, semantic segmentation, pose estimation, etc.</p>

<a href="https://colab.research.google.com/github/sithu31296/image-classification/blob/main/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

</div>

![visiontransformer](assests/vit_banner.png)

## <div align="center">Model Zoo</div>

[micronet]: https://arxiv.org/abs/2108.05894v1
[mobileformer]: https://arxiv.org/abs/2108.05895v1

[xcit]: https://arxiv.org/abs/2106.09681
[cswin]: https://arxiv.org/abs/2107.00652v2
[volo]: https://arxiv.org/abs/2106.13112v1
[gfnet]: https://arxiv.org/abs/2107.00645
[pvtv2]: https://arxiv.org/abs/2106.13797
[shuffle]: https://arxiv.org/abs/2106.03650
[conformer]: https://arxiv.org/abs/2105.03889v1
[rest]: https://arxiv.org/abs/2105.13677v3

[cyclemlp]: https://arxiv.org/abs/2107.10224
[hiremlp]: https://arxiv.org/abs/2108.13341
[smlp]: https://arxiv.org/abs/2109.05422
[poolformer]: https://arxiv.org/abs/2111.11418

Model | ImageNet-1k Top-1 Acc <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | Variants & Weights
--- | --- | --- | --- | --- 
[MicroNet][micronet] | 51.4`\|`59.4`\|`62.5 | 2`\|`2`\|`3 | 6M`\|`12M`\|`21M | [M1][micronetw]\|[M2][micronetw]\|[M3][micronetw]
||
[MobileFormer][mobileformer] | 76.7`\|`77.9`\|`79.3 | 9`\|`11`\|`14 | 214M`\|`294M`\|`508M | 214\|294\|508
||
[GFNet][gfnet] | 80.1`\|`81.5`\|`82.9 | 15`\|`32`\|`54 | 2`\|`5`\|`8 | [T][gfnett]\|[S][gfnets]\|[B][gfnetb]
[PVTv2][pvtv2] | 78.7`\|`82.0`\|`83.6 | 14`\|`25`\|`63 | 2`\|`4`\|`10 | [B1][pvt1]\|[B2][pvt2]\|[B4][pvt4]
[ResT][rest] | 79.6`\|`81.6`\|`83.6 | 14`\|`30`\|`52 | 2`\|`4`\|`8 | [S][rests]\|[B][restb]\|[L][restl]
||
[Conformer][conformer] | 81.3`\|`83.4`\|`84.1 | 24`\|`38`\|`83 | 5`\|`11`\|`23 | [T][conformert]\|[S][conformers]\|[B][conformerb]
[Shuffle][shuffle] | 82.4`\|`83.6`\|`84.0 | 28`\|`50`\|`88 | 5`\|`9`\|`16 | [T][shufflet]\|[S][shuffles]\|[B][shuffleb]
[CSWin][cswin] | 82.7`\|`83.6`\|`84.2 | 23`\|`35`\|`78 | 4`\|`7`\|`15 | [T][cswint]\|[S][cswins]\|[B][cswinb]
||
[CycleMLP][cyclemlp] | 81.6`\|`83.0`\|`83.2 | 27`\|`52`\|`76 | 4`\|`10`\|`12 | [B2][cycleb2]\|[B4][cycleb4]\|[B5][cycleb5]
[HireMLP][hiremlp] | 81.8`\|`83.1`\|`83.4 | 33`\|`58`\|`96 | 4`\|`8`\|`14 | S\|B\|L
[sMLP][smlp] | 81.9`\|`83.1`\|`83.4 | 24`\|`49`\|`66 | 5`\|`10`\|`14 | T\|S\|B
||
[PoolFormer][poolformer] | 80.3`\|`81.4`\|`82.1 | 21`\|`31`\|`56 | 4`\|`5`\|`9 | [S24][pfs24]\|[S36][pfs36]\|[M36][pfm36]
||
[XCiT][xcit] | 80.4`\|`83.9`\|`84.3 | 12`\|`48`\|`84 | 2`\|`9`\|`16 | [T][xcitt]\|[S][xcits]\|[M][xcitm]
[VOLO][volo] | 84.2`\|`85.2`\|`85.4 | 27`\|`59`\|`86 | 7`\|`14`\|`21 | [D1][volod1]\|[D2][volod2]\|[D3][volod3]

<details open>
  <summary><strong>Table Notes</strong></summary>

* Image size is 224x224. EfficientNetv2 uses progressive learning (image size from 128 to 380).
* All models' weights are from official repositories.
* Only models trained on ImageNet1k are compared. 
* (Parameters > 200M) Models are not included. 
* *PVTv2*, *ResT*, *Conformer*, *XCiT*, *CycleMLP* and *PoolFormer* models work with any image size.

</details>


## <div align="center">Usage</div>

<details>
  <summary><strong>Requirements</strong> (click to expand)</summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

</details>

<br>
<details open>
  <summary><strong>Show Available Models</strong></summary>

```bash
$ python tools/show.py
```

A table with model names and variants will be shown:

```
Model Names    Model Variants
-------------  --------------------------------
ResNet         ['18', '34', '50', '101', '152']
MicroNet       ['M1', 'M2', 'M3']
GFNet          ['T', 'S', 'B']
PVTv2          ['B1', 'B2', 'B3', 'B4', 'B5']
ResT           ['S', 'B', 'L']
Conformer      ['T', 'S', 'B']
Shuffle        ['T', 'S', 'B']
CSWin          ['T', 'S', 'B', 'L']
CycleMLP       ['B1', 'B2', 'B3', 'B4', 'B5']
XciT           ['T', 'S', 'M', 'L']
VOLO           ['D1', 'D2', 'D3', 'D4']
```

</details>

<details open>
  <summary><strong>Inference</strong></summary>

* Download your desired model's weights from `Model Zoo` table.
* Change `MODEL` parameters and `TEST` parameters in config file [here](./configs/test.yaml). And run the the following command.

```bash
$ python tools/infer.py --cfg configs/test.yaml
```

You will see an output similar to this:

```
File: assests\dog.jpg >>>>> Golden retriever
```
</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

```bash
$ python tools/train.py --cfg configs/train.yaml
```

</details>

<br>
<details>
  <summary><strong>Evaluate</strong> (click to expand)</summary>

```bash
$ python tools/val.py --cfg configs/train.yaml
```

</details>

<br>
<details>
  <summary><strong>Fine-tune</strong> (click to expand)</summary>

Fine-tune on CIFAR-10:

```bash
$ python tools/finetune.py --cfg configs/finetune.yaml
```

</details>

<br>
<details>
  <summary><strong>References</strong> (click to expand)</summary>

* https://github.com/rwightman/pytorch-image-models
* https://github.com/facebookresearch/deit

</details>

<br>
<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```
@article{zhql2021ResT,
  title={ResT: An Efficient Transformer for Visual Recognition},
  author={Zhang, Qinglong and Yang, Yubin},
  journal={arXiv preprint arXiv:2105.13677v3},
  year={2021}
}

@article{peng2021conformer,
  title={Conformer: Local Features Coupling Global Representations for Visual Recognition}, 
  author={Zhiliang Peng and Wei Huang and Shanzhi Gu and Lingxi Xie and Yaowei Wang and Jianbin Jiao and Qixiang Ye},
  journal={arXiv preprint arXiv:2105.03889},
  year={2021},
}

@misc{dong2021cswin,
  title={CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows}, 
  author={Xiaoyi Dong and Jianmin Bao and Dongdong Chen and Weiming Zhang and Nenghai Yu and Lu Yuan and Dong Chen and Baining Guo},
  year={2021},
  eprint={2107.00652},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{chen2021cyclemlp,
  title={CycleMLP: A MLP-like Architecture for Dense Prediction}, 
  author={Shoufa Chen and Enze Xie and Chongjian Ge and Ding Liang and Ping Luo},
  year={2021},
  eprint={2107.10224},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wang2021pvtv2,
  title={PVTv2: Improved Baselines with Pyramid Vision Transformer}, 
  author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  year={2021},
  eprint={2106.13797},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{elnouby2021xcit,
  title={XCiT: Cross-Covariance Image Transformers}, 
  author={Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Hervé Jegou},
  year={2021},
  eprint={2106.09681},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{yuan2021volo,
  title={VOLO: Vision Outlooker for Visual Recognition}, 
  author={Li Yuan and Qibin Hou and Zihang Jiang and Jiashi Feng and Shuicheng Yan},
  year={2021},
  eprint={2106.13112},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{yan2020micronet,
  title={MicroNet for Efficient Language Modeling}, 
  author={Zhongxia Yan and Hanrui Wang and Demi Guo and Song Han},
  year={2020},
  eprint={2005.07877},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{rao2021global,
  title={Global Filter Networks for Image Classification},
  author={Rao, Yongming and Zhao, Wenliang and Zhu, Zheng and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2107.00645},
  year={2021}
}

@article{huang2021shuffle,
  title={Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer},
  author={Huang, Zilong and Ben, Youcheng and Luo, Guozhong and Cheng, Pei and Yu, Gang and Fu, Bin},
  journal={arXiv preprint arXiv:2106.03650},
  year={2021}
}

```

</details>


[xcitt]: https://dl.fbaipublicfiles.com/xcit/xcit_tiny_24_p16_224_dist.pth
[xcits]: https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224_dist.pth
[xcitm]: https://dl.fbaipublicfiles.com/xcit/xcit_medium_24_p16_224_dist.pth
[cswint]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth
[cswins]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_small_224.pth
[cswinb]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_base_224.pth
[volod1]: https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar
[volod2]: https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar
[volod3]: https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar
[rests]: https://drive.google.com/file/d/18YGFK_ZqE_AXZ3cMLyM1Q-OnvWj0WlKZ/view?usp=sharing
[restb]: https://drive.google.com/file/d/1CdjkmikUM8tP6xKPGXXOlWdGJ9heIZqf/view?usp=sharing
[restl]: https://drive.google.com/file/d/1J60OCXwvlwbNiTwoRj-iLnGaAN9q0-g9/view?usp=sharing
[gfnett]: https://drive.google.com/file/d/1Nrq5sfHD9RklCMl6WkcVrAWI5vSVzwSm/view?usp=sharing
[gfnets]: https://drive.google.com/file/d/1w4d7o1LTBjmSkb5NKzgXBBiwdBOlwiie/view?usp=sharing
[gfnetb]: https://drive.google.com/file/d/1F900_-yPH7GFYfTt60xn4tu5a926DYL0/view?usp=sharing
[pvt1]: https://drive.google.com/file/d/1aM0KFE3f-qIpP3xfhihlULF0-NNuk1m7/view?usp=sharing
[pvt2]: https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing
[pvt4]: https://drive.google.com/file/d/1LW-0CFHulqeIxV2cai45t-FyLNKGc5l0/view?usp=sharing
[shufflet]: https://drive.google.com/drive/folders/1goDJtcnxgBAcHhZnNwrgOlG_WBftpmOS?usp=sharing
[shuffles]: https://drive.google.com/drive/folders/1GUBBQyDldY145vDiK-BHqivmpj3K6HK2?usp=sharing
[shuffleb]: https://drive.google.com/drive/folders/1x0biaJRdN4nxLmp_3lQcA_6hO_sDBoUM?usp=sharing
[cycleb2]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B2.pth
[cycleb4]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B4.pth
[cycleb5]: https://github.com/ShoufaChen/CycleMLP/releases/download/v0.1/CycleMLP_B5.pth
[conformert]: https://drive.google.com/file/d/19SxGhKcWOR5oQSxNUWUM2MGYiaWMrF1z/view?usp=sharing
[conformers]: https://drive.google.com/file/d/1mpOlbLaVxOfEwV4-ha78j_1Ebqzj2B83/view?usp=sharing
[conformerb]: https://drive.google.com/file/d/1oeQ9LSOGKEUaYGu7WTlUGl3KDsQIi0MA/view?usp=sharing
[micronetw]: https://drive.google.com/drive/folders/1j4JSTcAh94U2k-7jCl_3nwbNi0eduM2P?usp=sharing
[pfs24]: https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar
[pfs36]: https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar
[pfm36]: https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar