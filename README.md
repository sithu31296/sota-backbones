# <div align="center">SOTA Image Classification Models in PyTorch</div>

<div align="center">
<p>Intended for easy to use and integrate SOTA image classification models into object detection, semantic segmentation, pose estimation, etc.</p>

<a href="https://colab.research.google.com/github/sithu31296/image-classification/blob/main/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

</div>

![visiontransformer](assests/vit_banner.png)

## <div align="center">Model Zoo</div>

[micronet]: https://arxiv.org/abs/2108.05894v1
[mobileformer]: https://arxiv.org/abs/2108.05895v1

[cswin]: https://arxiv.org/abs/2107.00652v2
[gfnet]: https://arxiv.org/abs/2107.00645
[pvtv2]: https://arxiv.org/abs/2106.13797
[shuffle]: https://arxiv.org/abs/2106.03650
[conformer]: https://arxiv.org/abs/2105.03889v1
[rest]: https://arxiv.org/abs/2105.13677v3
[patchconvnet]: https://arxiv.org/abs/2112.13692

[cyclemlp]: https://arxiv.org/abs/2107.10224
[hiremlp]: https://arxiv.org/abs/2108.13341
[poolformer]: https://arxiv.org/abs/2111.11418
[rsb]: https://arxiv.org/abs/2110.00476
[wavemlp]: https://arxiv.org/abs/2111.12294
[convnext]: https://arxiv.org/abs/2201.03545

Model | ImageNet-1k Top-1 Acc <br><sup>(%) | Params <br><sup>(M)  | GFLOPs | Variants & Weights
--- | --- | --- | --- | --- 
[MicroNet][micronet] | 51.4`\|`59.4`\|`62.5 | 2`\|`2`\|`3 | 6M`\|`12M`\|`21M | [M1][micronetw]\|[M2][micronetw]\|[M3][micronetw]
[MobileFormer][mobileformer] | 76.7`\|`77.9`\|`79.3 | 9`\|`11`\|`14 | 214M`\|`294M`\|`508M | 214\|294\|508
||
[ResNet*][rsb] | 71.5`\|`80.4`\|`81.5 | 12`\|`26`\|`45 | 2`\|`4`\|`8 | [18][rsb18]\|[50][rsb50]\|[101][rsb101]
[ConvNeXt][convnext] | 82.1`\|`83.1`\|`83.8 | 28`\|`50`\|`89 | 5`\|`9`\|`15 | [T][convnextt]\|[S][convnexts]\|[B][convnextb]
||
[GFNet][gfnet] | 80.1`\|`81.5`\|`82.9 | 15`\|`32`\|`54 | 2`\|`5`\|`8 | [T][gfnett]\|[S][gfnets]\|[B][gfnetb]
[PVTv2][pvtv2] | 78.7`\|`82.0`\|`83.6 | 14`\|`25`\|`63 | 2`\|`4`\|`10 | [B1][pvt1]\|[B2][pvt2]\|[B4][pvt4]
[ResT][rest] | 79.6`\|`81.6`\|`83.6 | 14`\|`30`\|`52 | 2`\|`4`\|`8 | [S][rests]\|[B][restb]\|[L][restl]
||
[PoolFormer][poolformer] | 80.3`\|`81.4`\|`82.1 | 21`\|`31`\|`56 | 4`\|`5`\|`9 | [S24][pfs24]\|[S36][pfs36]\|[M36][pfm36]
[PatchConvnet][patchconvnet] | 82.1`\|`83.2`\|`83.5 | 25`\|`48`\|`99 | 4`\|`8`\|`16 | [S60][pcs60]\|[S120][pcs120]\|[B60][pcb60]
[Conformer][conformer] | 81.3`\|`83.4`\|`84.1 | 24`\|`38`\|`83 | 5`\|`11`\|`23 | [T][conformert]\|[S][conformers]\|[B][conformerb]
[Shuffle][shuffle] | 82.4`\|`83.6`\|`84.0 | 28`\|`50`\|`88 | 5`\|`9`\|`16 | [T][shufflet]\|[S][shuffles]\|[B][shuffleb]
[CSWin][cswin] | 82.7`\|`83.6`\|`84.2 | 23`\|`35`\|`78 | 4`\|`7`\|`15 | [T][cswint]\|[S][cswins]\|[B][cswinb]
||
[CycleMLP][cyclemlp] | 81.6`\|`83.0`\|`83.2 | 27`\|`52`\|`76 | 4`\|`10`\|`12 | [B2][cycleb2]\|[B4][cycleb4]\|[B5][cycleb5]
[HireMLP][hiremlp] | 79.7`\|`82.1`\|`83.2 | 18`\|`33`\|`58 | 2`\|`4`\|`8 | [T][hmlpt]\|[S][hmlps]\|[B][hmlpb]
[WaveMLP][wavemlp] | 80.9`\|`82.9`\|`83.3 | 17`\|`30`\|`44 | 2`\|`5`\|`8 | T\|S\|M

> Notes: ResNet* is from "ResNet strikes back" paper.

<details open>
  <summary><strong>Table Notes</strong></summary>

* Only include models trained on ImageNet1k with image size of 224x224.
* Models' weights are from respective official repositories.
* Large mdoels (Parameters > 100M) are not included. 

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
ConvNeXt       ['T', 'S', 'M']
GFNet          ['T', 'S', 'B']
PVTv2          ['B1', 'B2', 'B3', 'B4', 'B5']
ResT           ['S', 'B', 'L']
Conformer      ['T', 'S', 'B']
Shuffle        ['T', 'S', 'B']
CSWin          ['T', 'S', 'B', 'L']
CycleMLP       ['B1', 'B2', 'B3', 'B4', 'B5']
HireMLP        ['T', 'S', 'B']
WaveMLP        ['T', 'S', 'M']
PoolFormer     ['S24', 'S36', 'M36']
PatchConvnet   ['S60', 'S120', 'B60']
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
@misc{li2021micronet,
  title={MicroNet: Improving Image Recognition with Extremely Low FLOPs}, 
  author={Yunsheng Li and Yinpeng Chen and Xiyang Dai and Dongdong Chen and Mengchen Liu and Lu Yuan and Zicheng Liu and Lei Zhang and Nuno Vasconcelos},
  year={2021},
  eprint={2108.05894},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm}, 
  author={Ross Wightman and Hugo Touvron and Hervé Jégou},
  year={2021},
  eprint={2110.00476},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{rao2021global,
  title={Global Filter Networks for Image Classification}, 
  author={Yongming Rao and Wenliang Zhao and Zheng Zhu and Jiwen Lu and Jie Zhou},
  year={2021},
  eprint={2107.00645},
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

@misc{zhang2021rest,
  title={ResT: An Efficient Transformer for Visual Recognition}, 
  author={Qinglong Zhang and Yubin Yang},
  year={2021},
  eprint={2105.13677},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{touvron2021augmenting,
  title={Augmenting Convolutional networks with attention-based aggregation}, 
  author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Piotr Bojanowski and Armand Joulin and Gabriel Synnaeve and Hervé Jégou},
  year={2021},
  eprint={2112.13692},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{peng2021conformer,
  title={Conformer: Local Features Coupling Global Representations for Visual Recognition}, 
  author={Zhiliang Peng and Wei Huang and Shanzhi Gu and Lingxi Xie and Yaowei Wang and Jianbin Jiao and Qixiang Ye},
  year={2021},
  eprint={2105.03889},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{huang2021shuffle,
  title={Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer}, 
  author={Zilong Huang and Youcheng Ben and Guozhong Luo and Pei Cheng and Gang Yu and Bin Fu},
  year={2021},
  eprint={2106.03650},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{dong2022cswin,
  title={CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows}, 
  author={Xiaoyi Dong and Jianmin Bao and Dongdong Chen and Weiming Zhang and Nenghai Yu and Lu Yuan and Dong Chen and Baining Guo},
  year={2022},
  eprint={2107.00652},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{chen2021cyclemlp,
  title={CycleMLP: A MLP-like Architecture for Dense Prediction}, 
  author={Shoufa Chen and Enze Xie and Chongjian Ge and Runjian Chen and Ding Liang and Ping Luo},
  year={2021},
  eprint={2107.10224},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{guo2021hiremlp,
  title={Hire-MLP: Vision MLP via Hierarchical Rearrangement}, 
  author={Jianyuan Guo and Yehui Tang and Kai Han and Xinghao Chen and Han Wu and Chao Xu and Chang Xu and Yunhe Wang},
  year={2021},
  eprint={2108.13341},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision}, 
  author={Weihao Yu and Mi Luo and Pan Zhou and Chenyang Si and Yichen Zhou and Xinchao Wang and Jiashi Feng and Shuicheng Yan},
  year={2021},
  eprint={2111.11418},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{guo2021hiremlp,
  title={Hire-MLP: Vision MLP via Hierarchical Rearrangement}, 
  author={Jianyuan Guo and Yehui Tang and Kai Han and Xinghao Chen and Han Wu and Chao Xu and Chang Xu and Yunhe Wang},
  year={2021},
  eprint={2108.13341},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{tang2021image,
  title={An Image Patch is a Wave: Phase-Aware Vision MLP}, 
  author={Yehui Tang and Kai Han and Jianyuan Guo and Chang Xu and Yanxi Li and Chao Xu and Yunhe Wang},
  year={2021},
  eprint={2111.12294},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{liu2022convnet,
  title={A ConvNet for the 2020s}, 
  author={Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  year={2022},
  eprint={2201.03545},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

</details>


[cswint]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth
[cswins]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_small_224.pth
[cswinb]: https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_base_224.pth
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
[rsb18]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth
[rsb50]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
[rsb101]: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth
[pcs60]: https://dl.fbaipublicfiles.com/deit/s60_224_1k.pth
[pcs120]: https://dl.fbaipublicfiles.com/deit/s120_224_1k.pth
[pcb60]: https://dl.fbaipublicfiles.com/deit/b60_224_1k.pth
[hmlpt]: https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_tiny.pth
[hmlps]: https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_small.pth
[hmlpb]: https://github.com/ggjy/Hire-Wave-MLP.pytorch/releases/download/log/hire_mlp_base.pth
[convnextt]: https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
[convnexts]: https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
[convnextb]: https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth