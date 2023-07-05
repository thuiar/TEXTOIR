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
| Semi-supervised | [MTP-CLNN](./examples/run_DeepAligned.sh) | [Paper](https://aclanthology.org/2022.acl-long.21.pdf) [Code](https://github.com/LonelVino/MTP-CLNN) | ACL 2022 |
| Semi-supervised | [USNID](./examples/run_DeepAligned.sh) | [Paper](https://arxiv.org/pdf/2304.07699.pdf) [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) |  arXiv 2023 |


### Results
The detailed results can be seen in [results.md](results/results.md).
#### Overall Performance
* KIR means "Known Intent Ratio".  

| | | |BANKING     |  | | CLINC       |  |  |StackOverflow      |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR | Methods | NMI | ARI | ACC  |NMI | ARI | ACC | NMI | ARI | ACC  |
|0.0|KM|60.99|13.00|22.17|79.47|28.84|37.72|33.07|10.64|23.13|
|0.0|AG|62.85|14.41|23.96|80.33|27.99|36.54|33.59|10.24|22.18|
|0.0|SAE-KM|66.52|18.07|24.97|79.63|26.22|34.79|51.31|22.80|28.45|
|0.0|DEC|67.42|21.19|28.93|79.62|27.14|36.15|59.80|28.81|39.60|
|0.0|DCN|67.44|21.22|28.93|79.59|27.12|36.23|59.80|28.82|39.60|
|0.0|CC|56.41|8.48|16.35|76.07|18.60|28.71|19.06|8.79|21.01|
|0.0|SCCL|69.94|28.91|39.11|83.25|41.76|48.67|61.63|30.28|34.67|
|0.0|USNID|78.04|44.17|50.10|89.43|60.94|63.25|69.45|51.09|57.02|
| | | | | | | | | | | |  
|0.25|KCL|51.55|17.73|24.81|67.98|24.30|29.40|30.01|16.92|29.51|
|0.25|MCL|47.15|13.23|21.90|61.71|16.33|25.79|25.39|14.66|27.40|
|0.25|DTC|52.02|17.35|27.32|77.19|36.74|49.00|25.36|13.07|25.67|
|0.25|GCD|68.29|24.04|30.97|84.33|39.10|44.04|39.48|15.90|24.32|
|0.25|CDACPlus|71.46|26.28|32.33|84.06|35.49|40.87|47.39|21.15|26.77|
|0.25|DeepAligned|69.32|36.35|47.46|86.95|58.02|65.77|46.58|32.57|47.23|
|0.25|MTP-CLNN|79.26|33.87|34.62|87.43|39.73|40.21|65.43|45.44|48.14|
|0.25|USNID|82.77|55.79|60.86|92.65|71.59|71.83|75.59|65.32|69.76|
| | | | | | | | | | | |  
|0.5|KCL|63.22|29.95|39.64|74.64|35.01|45.25|53.38|41.71|55.72|
|0.5|MCL|62.61|29.12|41.02|76.46|37.89|47.61|45.63|38.67|50.77|
|0.5|DTC|69.90|36.76|49.40|83.07|50.51|64.24|48.45|34.98|50.07|
|0.5|GCD|72.02|30.20|36.52|86.09|44.79|47.90|52.54|25.38|30.69|
|0.5|CDACPlus|73.19|28.46|33.72|84.37|36.27|41.13|60.25|29.96|32.67|
|0.5|DeepAligned|75.43|45.60|56.11|90.97|68.38|70.99|54.84|35.80|47.43|
|0.5|MTP-CLNN|80.25|34.94|34.77|87.43|39.60|40.33|68.29|48.85|50.93|
|0.5|USNID|85.03|61.46|65.47|93.53|75.00|74.22|78.02|71.61|75.79|
| | | | | | | | | | | |  
|0.75|KCL|72.81|45.16|58.87|85.57|57.72|68.39|63.98|54.28|68.69|
|0.75|MCL|73.97|47.07|60.88|86.91|60.68|69.92|62.34|56.57|68.78|
|0.75|DTC|75.21|44.08|54.58|89.02|65.71|72.60|66.19|58.23|70.55|
|0.75|GCD|74.63|34.69|40.12|87.06|48.84|51.44|57.96|31.13|35.36|
|0.75|CDACPlus|74.01|29.39|34.33|84.43|36.49|41.32|58.79|28.91|31.21|
|0.75|DeepAligned|79.94|52.78|60.74|91.96|71.44|72.75|66.81|49.76|63.78|
|0.75|MTP-CLNN|80.49|35.49|35.25|87.67|40.56|40.99|67.90|45.07|47.08|
|0.75|USNID|86.60|65.86|68.97|93.82|75.71|74.10|78.83|72.89|77.09|












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
* [USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery](https://arxiv.org/abs/2304.07699)

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
@article{USNID, 
    title={USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery},  
    author={Zhang, Hanlei and Xu, Hua and Wang, Xin and Long, Fei and Gao, Kai},
    journal={arXiv preprint arXiv:2304.07699},  
    year={2023}, 
 } 
```