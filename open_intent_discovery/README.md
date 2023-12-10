# Open Intent Discovery

This package provides the toolkit for open intent discovery implemented with PyTorch (for semi-supervised clustering methods), and Tensorflow (for unsupervised clustering methods).

## Introduction

Open intent discovery aims to leverage limited labeled data of known intents to help find discover open intent clusters. We regard it as a clustering problem, and classifies the related methods into two categories, semi-supervised clustering (with some labeled known intent data as prior knowledge), and unsupervised clustering (without any prior knowledge). An example is as follows:

<img src="figs/open_intent_discovery.png" width="340" height = "200">

We collect benchmark intent datasets, and reproduce related methods to our best. For the convenience of users, we provide flexible and extensible interfaces to add new methods. Welcome to contact us (zhang-hl20@mails.tsinghua.edu.cn) to add your methods!

## Basic Information

### Benchmark Datasets
| Dataset Name | Source |
| :---: | :---: |
| [BANKING](../data/banking) | [Paper](https://aclanthology.org/2020.nlp4convai-1.5/) |
| [CLINC150](../data/clinc) | [Paper](https://aclanthology.org/D19-1131/) |
| [StackOverflow](../data/stackoverflow) | [Paper](https://aclanthology.org/W15-1509.pdf) |
### Integrated Models

| Setting | Model Name | Source | Published |
| :---: | :---: | :---: | :---: |
| Unsupervised | [KM](./examples/run_KM.sh) | [Paper](https://www.cs.cmu.edu/~bhiksha/courses/mlsp.fall2010/class14/macqueen.pdf) | BSMSP 1967 |
| Unsupervised | [AG](./examples/run_AG.sh) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320378900183) | PR 1978 |
| Unsupervised | [SAE-KM](./examples/run_SAE.sh) | [Paper](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)  | JMLR 2010|
| Unsupervised | [DEC](./examples/run_DEC.sh) | [Paper](http://proceedings.mlr.press/v48/xieb16.pdf) [Code](https://github.com/piiswrong/dec) | ICML 2016 |
| Unsupervised | [DCN](./examples/run_DCN.sh) | [Paper](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) [Code](https://github.com/xuyxu/Deep-Clustering-Network) | ICML 2017 |
| Unsupervised | [CC](./examples/run_DCN.sh) | [Paper](https://yunfan-li.github.io/assets/pdf/Contrastive%20Clustering.pdf) [Code](https://github.com/Yunfan-Li/Contrastive-Clustering) | AAAI 2021 |
| Unsupervised | [SCCL](./examples/run_DCN.sh) | [Paper](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) [Code](https://github.com/xuyxu/Deep-Clustering-Network) | NAACL 2021 |
| Unsupervised | [USNID](./examples/run_DCN.sh) | [Paper](https://arxiv.org/pdf/2304.07699.pdf) [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) |   arXiv 2023 |
| Semi-supervised | [KCL*](./examples/run_KCL.sh) | [Paper](https://openreview.net/pdf?id=ByRWCqvT-) [Code](https://github.com/GT-RIPL/L2C) | ICLR 2018 |
| Semi-supervised | [MCL*](./examples/run_MCL.sh) | [Paper](https://openreview.net/pdf?id=SJzR2iRcK7) [Code](https://github.com/GT-RIPL/L2C) | ICLR 2019 |
| Semi-supervised | [DTC*](./examples/run_DTC.sh) | [Paper](https://www.robots.ox.ac.uk/~vgg/research/DTC/files/iccv2019_DTC.pdf) [Code](https://github.com/k-han/DTC) | ICCV 2019 |
| Semi-supervised | [CDAC+](./examples/run_CDACPlus.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6353) [Code](https://github.com/thuiar/CDAC-plus) | AAAI 2020 |
| Semi-supervised | [DeepAligned](./examples/run_DeepAligned.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17689) [Code](https://github.com/thuiar/DeepAligned-Clustering) | AAAI 2021 |
| Semi-supervised | [GCD](./examples/run_DeepAligned.sh) | [Paper](https://www.robots.ox.ac.uk/~vgg/research/gcd/) [Code](https://github.com/sgvaze/generalized-category-discovery) | CVPR 2022 |
| Semi-supervised | [MTP-CLNN](./examples/run_DeepAligned.sh) | [Paper](https://aclanthology.org/2022.acl-long.21.pdf) [Code](https://github.com/fanolabs/NID_ACLARR2022) | ACL 2022 |
| Semi-supervised | [USNID](./examples/run_DeepAligned.sh) | [Paper](https://arxiv.org/pdf/2304.07699.pdf) [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) |  arXiv 2023 |


### Results
The detailed results can be seen in [results.md](results/results.md).
#### Overall Performance
* KIR means "Known Intent Ratio".  

| | | |BANKING     |  | | CLINC       |  |  |StackOverflow      |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR | Methods | NMI | ARI | ACC  |NMI | ARI | ACC | NMI | ARI | ACC  |
|0.0|KM|49.30|13.04|28.62|71.05|27.72|45.76|19.87|5.23|23.72|
|0.0|AG|53.28|14.65|31.62|72.21|27.05|44.12|25.54|7.12|28.50|
|0.0|SAE-KM|59.80|23.59|37.07|73.77|31.58|47.15|44.96|28.23|49.11|
|0.0|DEC|62.66|25.32|38.60|74.83|31.71|48.77|58.76|36.23|59.49|
|0.0|DCN|62.72|25.36|38.59|74.77|31.68|48.69|58.75|36.23|59.48|
|0.0|CC|44.89|9.75|21.51|65.79|18.00|32.69|19.06|8.79|21.01|
|0.0|SCCL|63.89|26.98|40.54|79.35|38.14|50.44|69.11|34.81|68.15|
|0.0|USNID|**75.30**|**43.33**|**54.82**|**91.00**|**68.54**|**75.87**|**72.00**|**52.25**|**69.28**|
| | | | | | | | | | | | 
|0.25|KCL|52.70|18.58|26.03|67.98|24.30|29.40|30.42|17.66|30.69|
|0.25|MCL|47.88|14.43|23.29|62.76|18.21|28.52|26.68|17.54|31.46|
|0.25|DTC|55.59|19.09|31.75|79.35|41.92|56.90|29.96|17.51|29.54|
|0.25|GCD|60.89|27.30|39.91|83.69|52.13|64.69|31.72|16.81|36.76|
|0.25|CDACPlus|66.39|33.74|48.00|84.68|50.02|66.24|46.16|30.99|51.61|
|0.25|DeepAligned|70.50|37.62|49.08|88.97|64.63|74.07|50.86|37.96|54.50|
|0.25|MTP-CLNN|80.04|52.91|65.06|93.17|76.20|**83.26**|73.35|54.80|74.70|
|0.25|USNID|**81.94**|**56.53**|**65.85**|**94.17**|**77.95**|83.12|**74.91**|**65.45**|**75.76**|
| | | | | | | | | | | | 
|0.5|KCL|63.50|30.36|40.04|74.74|35.28|45.69|53.39|41.74|56.80|
|0.5|MCL|62.71|29.91|41.94|76.94|39.74|49.44|45.17|36.28|52.53|
|0.5|DTC|69.46|37.05|49.85|83.01|50.45|64.39|49.80|37.38|52.92|
|0.5|GCD|67.29|35.52|48.37|87.12|59.75|70.93|49.57|31.15|53.77|
|0.5|CDACPlus|67.30|34.97|48.55|86.00|54.87|68.01|46.21|30.88|51.79|
|0.5|DeepAligned|76.67|47.95|59.38|91.59|72.56|80.70|68.28|57.62|74.52|
|0.5|MTP-CLNN|83.42|60.17|70.97|94.30|80.17|86.18|76.66|62.24|80.36|
|0.5|USNID|**85.05**|**63.77**|**73.27**|**95.48**|**82.99**|**87.28**|**78.77**|**71.63**|**82.06**|
| | | | | | | | | | | | 
|0.75|KCL|72.75|45.21|59.12|86.01|58.62|68.89|63.98|54.28|68.69|
|0.75|MCL|74.42|48.06|61.56|87.26|61.21|70.27|63.44|56.11|71.71|
|0.75|DTC|74.44|44.68|57.16|89.19|67.15|77.65|63.05|53.83|71.04|
|0.75|GCD|72.21|42.86|56.94|89.38|66.03|76.82|60.14|42.05|65.20|
|0.75|CDACPlus|69.54|37.78|51.07|85.96|55.17|67.77|58.23|40.95|64.57|
|0.75|DeepAligned|79.39|53.09|64.63|93.92|79.94|86.79|73.28|60.09|77.97|
|0.75|MTP-CLNN|86.19|66.98|77.22|95.45|84.30|89.46|77.12|69.36|82.90|
|0.75|USNID|**87.41**|**69.54**|**78.36**|**96.42**|**86.77**|**90.36**|**80.13**|**74.90**|**85.66**|

We welcome any issues and requests for model implementation and bug fix. 

### Data Settings

Each dataset is split to training, development, and testing sets. We select partial intents as known (the labeled ratio can be changed) intents. Notably, we uniformly select 10% as labeled from known intent data. We use all training data (both labeled and unlabeled) to train the model. During testing, we evaluate the clustering performance of all intent classes. More detailed information can be seen in the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17689).

### Parameter Configurations

The basic parameters include parsing parameters about selected dataset, method, setting, etc. More details can be seen in [run.py](./run.py). For specific parameters of each method, we support add configuration files with different hyper-parameters in the [configs](./configs) directory. 

An example can be seen in [DeepAligned.py](./configs/DeepAligned.py). Notice that the config file name is corresponding to the parsing parameter.

Normally, the input commands are as follows:
```
python run.py --setting xxx --dataset xxx --known_cls_ratio xxx --labeled_ratio xxx --cluster_num_factor xxx --config_file_name xxx
```

Notice that if you want to train the model, save the model, or save the testing results, you need to add related parameters (--train, --save_model, --save_results)

## Tutorials
### a. How to add a new dataset? 
1. Prepare Data  
Create a new directory to store your dataset in the [data](../data) directory. You should provide the train.tsv, dev.tsv, and test.tsv, with the same formats as in the provided [datasets](../data/banking).

2. Dataloader Setting  
Calculate the maximum sentence length (token unit) and count the labels of the dataset. Add them in the [file](./dataloaders/init.py) as follows:  
```
max_seq_lengths = {
    'new_dataset': max_length
}
benchmark_labels = {
    'new_dataset': label_list
}
```

### b. How to add a new backbone?

1. Add a new backbone in the [backbones](./backbones) directory. For example, we provide [bert-based](./backbones/bert.py), [glove-based](./backbones/glove.py), and [sae-based](./backbones/sae.py) backbones.

2. Add the new backbone mapping in the [file](./backbones/__init__.py) as follows:
```
from .bert import new_backbone_class
backbones_map = {
    'new_backbone': new_backbone_class
}
```
Add a new loss in the [losses](./losses) directory is almost the same as adding a new backbone.

### c. How to add a new method?

1. Configuration Setting   
Create a new file, named "method_name.py" in the [configs](./configs) directory, and set the hyper-parameters for the method (an example can be seen in [DeepAligned.py](./configs/DeepAligned.py)). 

2. Dataloader Setting  
Add the dataloader mapping if you use new backbone for the method. For example, the bert-based model corresponds to the bert dataloader as follows.
```
from .bert_loader import BERT_Loader
backbone_loader_map = {
    'bert': BERT_Loader,
    'bert_xxx': BERT_Loader
}
```
The unsupervised clustering methods use the unified dataloader as follows:
```
from .unsup_loader import UNSUP_Loader
backbone_loader_map = {
    'glove': UNSUP_Loader,
    'sae': UNSUP_Loader
}
```

3. Add Methods  (Take DeepAligned as an example)

- Classify the method into the corresponding category in the [methods](./methods) directory. For example, DeepAligned belongs to the [semi-supervised](./methods/semi_supervised) directory, and creates a subdirectory under it, named "DeepAligned". 

- Add the manager file for DeepAligned. The file should include the method manager class (e.g., DeepAlignedManager), which includes training, evalutation, and testing modules for the method. An example can be seen in [manager.py](./methods/semi_supervised/DeepAligned/manager.py).  

- Add the related method dependency in [__init__.py](./methods/__init__.py) as below:
```
from .semi_supervised.DeepAligned.manager import DeepAlignedManager
method_map = {
    'DeepAligned': DeepAlignedManager
}
```
(The key corresponds to the input parameter "method")

4. Run Examples
Add a script in the [examples](./examples) directory, and configure the parsing parameters in the [run.py](./run.py). You can also run the programs serially by setting the combination of different parameters. A running example is shown in [run_DeepAligned.sh](./examples/run_DeepAligned.sh).

## Citations
If you are interested in this work, and want to use the codes in this repo, please star/fork this repo, and cite the following works:

* [TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition](https://aclanthology.org/2021.acl-demo.20/)
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
@ARTICLE{10349963,
  author={Zhang, Hanlei and Xu, Hua and Wang, Xin and Long, Fei and Gao, Kai},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Clustering Framework for Unsupervised and Semi-supervised New Intent Discovery}, 
  year={2023},
  doi={10.1109/TKDE.2023.3340732}
}
```
