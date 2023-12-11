# TEXT Open Intent Recognition (TEXTOIR)

TEXTOIR is the first high-quality Text Open Intent Recognition platform. This repo contains a convenient toolkit with extensible interfaces, integrating a series of state-of-the-art algorithms of two tasks (open intent detection and open intent discovery). We also release the pipeline framework and the visualized platform in the repo [TEXTOIR-DEMO](https://github.com/thuiar/TEXTOIR-DEMO). 


## Introduction
TEXTOIR aims to provide a convenience toolkit for researchers to reproduce the related text open classification and clustering methods. It contains two tasks, which are defined as open intent detection and open intent discovery. Open intent detection aims to identify n-class known intents, and detect one-class open intent. Open intent discovery aims to leverage limited prior knowledge of known intents to find fine-grained known and open intent-wise clusters. Related papers and codes are collected in our previous released [reading list](https://github.com/thuiar/OKD-Reading-List).

Open Intent Recognition:  
![Example](figs/Intro.png "Example")

## Updates 🔥 🔥 🔥 

| Date 	| Announcements 	|
|-	|-	|
| 12/2023  | 🎆 🎆 New paper and SOTA in Open Intent Discovery. Refer to the directory [USNID](./open_intent_discovery/examples/run_semi_usnid.sh) for the codes. Read the paper -- [A Clustering Framework for Unsupervised and Semi-supervised New Intent Discovery (Published in IEEE TKDE 2023)](https://ieeexplore.ieee.org/document/10349963).  |
| 04/2023  | 🎆 🎆 New paper and SOTA in Open Intent Detection. Refer to the directory [DA-ADB](./open_intent_detection/examples/run_DA-ADB.sh) for the codes. Read the paper -- [Learning Discriminative Representations and Decision Boundaries for Open Intent Detection (Published in IEEE/ACM TASLP 2023)](https://ieeexplore.ieee.org/document/10097558).  |
| 09/2021 	| 🎆 🎆 The first integrated and visualized platform for text Open Intent Recognition TEXTOIR has been released. Refer to the directory [TEXTOIR-DEMO](https://github.com/thuiar/TEXTOIR-DEMO) for the demo codes. Read our paper [TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition (Published in ACL 2021)](https://aclanthology.org/2021.acl-demo.20.pdf).	|
| 05/2021 	| New paper and baselines DeepAligned in Open Intent Discovery have been released. Read our paper [Discovering New Intents with Deep Aligned Clustering (Published in AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17689). 	|
| 05/2021 	| New paper and baselines ADB in Open Intent Detection have been released. Read our paper [Deep Open Intent Classification with Adaptive Decision Boundary (Published in AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17690). 	|
| 05/2020 	| New paper and baselines CDAC+ in Open Intent Discovery have been released. Read our paper [Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement (Published in AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6353). 	|
---------------------------------------------------------------------------

 **We strongly recommend you to use our TEXTOIR toolkit, which has standard and unified interfaces (especially data setting) to obtain fair and persuable results on benchmark intent datasets!**
 
## Benchmark Datasets

| Datasets | Source |
| :---: | :---: |
| [BANKING](./data/banking) | [Paper](https://aclanthology.org/2020.nlp4convai-1.5/) |
| [OOS](./data/oos) / [CLINC150](./data/clinc) | [Paper](https://aclanthology.org/D19-1131/) |
| [StackOverflow](./data/stackoverflow) | [Paper](https://aclanthology.org/W15-1509.pdf) |

## Integrated Models
### Open Intent Detection

| Model Name | Source | Published |
| :---: | :---: | :---: |
| [OpenMax*](./open_intent_detection/examples/run_OpenMax.sh) | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) [Code](https://github.com/abhijitbendale/OSDN) | CVPR 2016 |
| [MSP](./open_intent_detection/examples/run_MSP.sh) | [Paper](https://arxiv.org/pdf/1610.02136.pdf) [Code](https://github.com/hendrycks/error-detection) | ICLR 2017 |
| [DOC](./open_intent_detection/examples/run_DOC.sh) | [Paper](https://aclanthology.org/D17-1314.pdf) [Code](https://github.com/leishu02/EMNLP2017_DOC) | EMNLP 2017 |
| [DeepUnk](./open_intent_detection/examples/run_DeepUnk.sh) | [Paper](https://aclanthology.org/P19-1548.pdf) [Code](https://github.com/thuiar/DeepUnkID) | ACL 2019 |
| [SEG](./open_intent_detection/examples/run_SEG.sh) | [Paper](https://aclanthology.org/2020.acl-main.99) [Code](https://github.com/fanolabs/0shot-classification) | ACL 2020 |
| [ADB](./open_intent_detection/examples/run_ADB.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17690) [Code](https://github.com/thuiar/Adaptive-Decision-Boundary) | AAAI 2021 |
| [(K+1)-way](./open_intent_detection/examples/run_K+1-way.sh) | [Paper](https://aclanthology.org/2021.acl-long.273) [Code](https://github.com/fanolabs/out-of-scope-intent-detection) | ACL 2021 |
| [MDF](./open_intent_detection/examples/run_MDF.sh) | [Paper](https://aclanthology.org/2021.acl-long.85.pdf) [Code](https://github.com/rivercold/BERT-unsupervised-OOD) | ACL 2021 |
| [ARPL*](./open_intent_detection/examples/run_ARPL.sh) | [Paper](https://ieeexplore.ieee.org/document/9521769) [Code](https://github.com/iCGY96/ARPL) | IEEE TPAMI 2022 |
| [KNNCL](./open_intent_detection/examples/run_KNNCL.sh) | [Paper](https://aclanthology.org/2022.acl-long.352/) [Code](https://github.com/zyh190507/KnnContrastiveForOOD) | ACL 2022 |
| [DA-ADB](./open_intent_detection/examples/run_DA-ADB.sh) | [Paper](https://ieeexplore.ieee.org/document/10097558) [Code](https://github.com/thuiar/TEXTOIR) | IEEE/ACM TASLP 2023 |

### Open Intent Discovery

| Setting | Model Name | Source | Published |
| :---: | :---: | :---: | :---: |
| Unsupervised | [KM](./examples/run_KM.sh) | [Paper](https://www.cs.cmu.edu/~bhiksha/courses/mlsp.fall2010/class14/macqueen.pdf) | BSMSP 1967 |
| Unsupervised | [AG](./examples/run_AG.sh) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320378900183) | PR 1978 |
| Unsupervised | [SAE-KM](./examples/run_SAE.sh) | [Paper](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)  | JMLR 2010|
| Unsupervised | [DEC](./examples/run_DEC.sh) | [Paper](http://proceedings.mlr.press/v48/xieb16.pdf) [Code](https://github.com/piiswrong/dec) | ICML 2016 |
| Unsupervised | [DCN](./examples/run_DCN.sh) | [Paper](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) [Code](https://github.com/xuyxu/Deep-Clustering-Network) | ICML 2017 |
| Unsupervised | [CC](./examples/run_CC.sh) | [Paper](https://yunfan-li.github.io/assets/pdf/Contrastive%20Clustering.pdf) [Code](https://github.com/Yunfan-Li/Contrastive-Clustering) | AAAI 2021 |
| Unsupervised | [SCCL](./examples/run_SCCL.sh) | [Paper](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) [Code](https://github.com/xuyxu/Deep-Clustering-Network) | NAACL 2021 |
| Unsupervised | [USNID](./examples/run_unsup_usnid) | [Paper](https://ieeexplore.ieee.org/document/10349963) [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) |   IEEE TKDE 2023 |
| Semi-supervised | [KCL*](./examples/run_KCL.sh) | [Paper](https://openreview.net/pdf?id=ByRWCqvT-) [Code](https://github.com/GT-RIPL/L2C) | ICLR 2018 |
| Semi-supervised | [MCL*](./examples/run_MCL.sh) | [Paper](https://openreview.net/pdf?id=SJzR2iRcK7) [Code](https://github.com/GT-RIPL/L2C) | ICLR 2019 |
| Semi-supervised | [DTC*](./examples/run_DTC.sh) | [Paper](https://www.robots.ox.ac.uk/~vgg/research/DTC/files/iccv2019_DTC.pdf) [Code](https://github.com/k-han/DTC) | ICCV 2019 |
| Semi-supervised | [CDAC+](./examples/run_CDACPlus.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6353) [Code](https://github.com/thuiar/CDAC-plus) | AAAI 2020 |
| Semi-supervised | [DeepAligned](./examples/run_DeepAligned.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17689) [Code](https://github.com/thuiar/DeepAligned-Clustering) | AAAI 2021 |
| Semi-supervised | [GCD](./examples/run_GCD.sh) | [Paper](https://www.robots.ox.ac.uk/~vgg/research/gcd/) [Code](https://github.com/sgvaze/generalized-category-discovery) | CVPR 2022 |
| Semi-supervised | [MTP-CLNN](./examples/run_MTP_CLNN.sh) | [Paper](https://aclanthology.org/2022.acl-long.21.pdf) [Code](https://github.com/fanolabs/NID_ACLARR2022) | ACL 2022 |
| Semi-supervised | [USNID](./examples/run_semi_usnid.sh) | [Paper](https://ieeexplore.ieee.org/document/10349963) [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) |  IEEE TKDE 2023 |


(* denotes the CV model replaced with the BERT backbone)

## Quick Start
1. Use anaconda to create Python (version >= 3.6) environment
```
conda create --name textoir python=3.6
conda activate textoir
```
2. Install PyTorch (Cuda version 11.2)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge  
```
2. Clone the TEXTOIR repository, and choose the task (Take open intent detection as an example).
```
git clone git@github.com:thuiar/TEXTOIR.git
cd TEXTOIR
cd open_intent_detection
```
3. Install related environmental dependencies
```
pip install -r requirements.txt
```
4. Run examples (Take ADB as an example)
```
sh examples/run_ADB.sh
```

## Extensibility

This toolkit is extensible and supports adding new methods, datasets, configurations, backbones, dataloaders, losses conveniently. More detailed information can be seen in the directory [open_intent_detection](./open_intent_detection/README.md) and [open_intent_discovery](./open_intent_discovery/README.md) respectively. 

<!-- ### Extensibility
This toolkit is extensible and supports adding new methods, datasets, configurations, backbones, dataloaders, losses conveniently. More detailed information can be seen in the directory [open_intent_detection](./open_intent_detection/README.md) and [open_intent_discovery](./open_intent_discovery/README.md) respectively. 

### Reliability
The codes in this repo have been confirmed and are reliable. 

The experimental results are close to the reported ones in our AAAI 2021 papers [Discovering New Intents with DeepAligned Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/17689) and [Deep Open Intent Classification with Adaptive Decision Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/17690). Note that the results of some methods may fluctuate in a small range due to the selected random seeds, hyper-parameters, optimizers, etc. The final results are the average of 10 random seeds to reduce the influence of different selected known classes. -->

## Citations

If this work is helpful, or you want to use the codes and results in this repo, please cite the following papers:

* [TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition](https://aclanthology.org/2021.acl-demo.20/)
* [Learning Discriminative Representations and Decision Boundaries for Open Intent Detection](https://ieeexplore.ieee.org/document/10097558)
* [A Clustering Framework for Unsupervised and Semi-supervised New Intent Discovery](https://ieeexplore.ieee.org/document/10349963)

```
@inproceedings{zhang-etal-2021-textoir,
    title = "{TEXTOIR}: An Integrated and Visualized Platform for Text Open Intent Recognition",
    author = "Zhang, Hanlei  and Li, Xiaoteng  and Xu, Hua  and Zhang, Panpan and Zhao, Kang  and Gao, Kai",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    pages = "167--174",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-demo.20",
    doi = "10.18653/v1/2021.acl-demo.20",
}
```
```
@article{DA-ADB, 
    title = {Learning Discriminative Representations and Decision Boundaries for Open Intent Detection},  
    author = {Zhang, Hanlei and Xu, Hua and Zhao, Shaojie and Zhou, Qianrui}, 
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},  
    volume = {31},
    pages = {1611-1623},
    year = {2023}, 
    doi = {10.1109/TASLP.2023.3265203} 
} 
```
```
@ARTICLE{USNID,
  author={Zhang, Hanlei and Xu, Hua and Wang, Xin and Long, Fei and Gao, Kai},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Clustering Framework for Unsupervised and Semi-supervised New Intent Discovery}, 
  year={2023},
  doi={10.1109/TKDE.2023.3340732}
} 
```


## Contributors

[Hanlei Zhang](https://github.com/HanleiZhang), [Shaojie Zhao](https://github.com/MurraryZhao), [Xin Wang](https://github.com/mrFocusXin), [Ting-En Lin](https://github.com/tnlin), [Qianrui Zhou](https://github.com/zhougr18), [Huisheng Mao](https://github.com/FlameSky-S). 

## Bugs or questions?

If you have any questions, please open issues and illustrate your problems as detailed as possible. If you want to integrate your method in our repo, please feel free to **pull request**!
