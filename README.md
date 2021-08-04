# TEXTOIR
TEXTOIR is an Integrated and Extensible Platform for Text Open Intent Recognition. 

If you are insterested in this work, and want to use the codes or results in this repository, please **star** and **fork** this repository and **cite** by:
```
@inproceedings{zhang-etal-2021-textoir,
    title = "{TEXTOIR}: An Integrated and Visualized Platform for Text Open Intent Recognition",
    author = "Zhang, Hanlei  and
      Li, Xiaoteng  and
      Xu, Hua  and
      Zhang, Panpan  and
      Zhao, Kang  and
      Gao, Kai",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    year = "2021",
    pages = "167--174",
}
```

If you have any questions, feel free to open issues, pull request, or contact us (zhang-hl20@mails.tsinghua.edu.cn). Please illustrate your problems as detailed as possible. 

## Introduction
TEXTOIR aims to provide a convenience toolkit for researchers to reproduce the related text open classification and clustering methods. It contains two tasks, which are defined as open intent detection and open intent discovery. Open intent detection aims to identify n-class known intents, and detect one-class open intent. Open intent discovery aims to leverage limited prior knowledge of known intents to find fine-grained known and open intent-wise clusters.

![](figs/Intro.png "Open Intent Detection and Discovery:")
## Benmark Datasets
* [BANKING](https://arxiv.org/pdf/2003.04807.pdf)
* [OOS / CLINC150 (without OOD samples)](https://arxiv.org/pdf/1909.02027.pdf) 
* [StackOverflow](https://aclanthology.org/W15-1509.pdf)

 **We strongly recommend you to use the framework with standard and unified interfaces (especially data setting) to obtain fair and persuable results on benchmark intent datasets!**

## Integrated Models
### Open Intent Detection

* [Deep Open Intent Classification with Adaptive Decision Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/17690) (ADB)
* [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (DeepUnk)
* [DOC: Deep Open Classification of Text Documents](https://aclanthology.org/D17-1314.pdf) (DOC)
* [A Baseline For Detecting Misclassified and Out-of-distribution Examples in Neural Networks](https://arxiv.org/pdf/1610.02136.pdf) (MSP) 
* [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (OpenMax)


### Open Intent Discovery

* Semi-supervised Clustering Methods
    - [Discovering New Intents with Deep Aligned Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/17689) (DeepAligned)
    - [Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement](https://ojs.aaai.org/index.php/AAAI/article/view/6353) (CDACPlus)
    - [Learning to Discover Novel Visual Categories via Deep Transfer Clustering](https://www.robots.ox.ac.uk/~vgg/research/DTC/files/iccv2019_DTC.pdf) (DTC*)
    - [Multi-class Classification Without Multi-class Labels](https://openreview.net/pdf?id=SJzR2iRcK7) (MCL*)
    - [Learning to cluster in order to transfer across domains and tasks](https://openreview.net/pdf?id=ByRWCqvT-) (KCL*)
* Unsupervised Clustering Methods
    - [Deep Clustering Network](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) (DCN)
    - [Deep Embedded Clustering](http://proceedings.mlr.press/v48/xieb16.pdf) (DEC)
    - Stacked auto-encoder K-Means (SAE-KM)
    - Agglomerative clustering (AG)
    - K-Means (KM)

(* denotes the CV model replaced with the BERT backbone)