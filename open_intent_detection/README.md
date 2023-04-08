# Open Intent Detection

This package provides the toolkit for open intent detection implemented with PyTorch.

## Introduction

Open intent detection aims to identify n-class known intents, and detect the one-class open intent, which is regarded as an open classification problem. The following is an example:

<img src="figs/open_intent_detection.png" width="360" height = "200">

We collect benchmark intent datasets, and reproduce related methods to our best. For the convenience of users, we provide flexible and extensible interfaces to add new methods. Welcome to contact us (zhang-hl20@mails.tsinghua.edu.cn) to add your methods!

## Basic Information

### Benchmark Datasets
| Datasets | Source |
| :---: | :---: |
| [BANKING](../data/banking) | [Paper](https://aclanthology.org/2020.nlp4convai-1.5/) |
| [OOS](../data/oos) | [Paper](https://aclanthology.org/D19-1131/) |
| [StackOverflow](../data/stackoverflow) | [Paper](https://aclanthology.org/W15-1509.pdf) |

### Integrated Models

| Model Name | Source | Published |
| :---: | :---: | :---: |
| [DA-ADB](./examples/run_DA-ADB.sh) | [Paper](https://ieeexplore.ieee.org/document/10097558) [Code](https://github.com/thuiar/TEXTOIR) | IEEE/ACM TASLP 2023 |
| [ARPL*](./examples/run_ARPL.sh) | [Paper](https://ieeexplore.ieee.org/document/9521769) [Code](https://github.com/iCGY96/ARPL) | IEEE TPAMI 2022 |
| [MDF](./examples/run_MDF.sh) | [Paper](https://aclanthology.org/2021.acl-long.85.pdf) [Code](https://github.com/rivercold/BERT-unsupervised-OOD) | ACL 2021 |
| [(K+1)-way](./examples/run_K+1-way.sh) | [Paper](https://aclanthology.org/2021.acl-long.273) [Code](https://github.com/fanolabs/out-of-scope-intent-detection) | ACL 2021 |
| [ADB](./examples/run_ADB.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17690) [Code](https://github.com/thuiar/Adaptive-Decision-Boundary) | AAAI 2021 |
| [SEG](./examples/run_SEG.sh) | [Paper](https://aclanthology.org/2020.acl-main.99) [Code](https://github.com/fanolabs/0shot-classification) | ACL 2020 |
| [DeepUnk](./examples/run_DeepUnk.sh) | [Paper](https://aclanthology.org/P19-1548.pdf) [Code](https://github.com/thuiar/DeepUnkID) | ACL 2019 |
| [DOC](./examples/run_DOC.sh) | [Paper](https://aclanthology.org/D17-1314.pdf) [Code](https://github.com/leishu02/EMNLP2017_DOC) | EMNLP 2017 |
| [MSP](./examples/run_MSP.sh) | [Paper](https://arxiv.org/pdf/1610.02136.pdf) [Code](https://github.com/hendrycks/error-detection) | ICLR 2017 |
| [OpenMax*](./examples/run_OpenMax.sh) | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) [Code](https://github.com/abhijitbendale/OSDN) | CVPR 2016 |

We welcome any issues and requests for model implementation and bug fix. 

### Data Settings

Each dataset is split to training, development, and testing sets. We select partial intents as known (the labeled ratio can be changed) for training, and use all intents for testing. All the unknown intents are regarded as one open class (with token \<UNK> or \<OOS> in our codes). More detailed information can be seen in the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17690).

### Parameter Configurations

The basic parameters include parsing parameters about selected dataset, method, setting, etc. More details can be seen in [run.py](./run.py). For specific parameters of each method, we support add configuration files with different hyper-parameters in the [configs](./configs) directory. 

An example can be seen in [ADB.py](./configs/ADB.py). Notice that the config file name is corresponding to the parsing parameter.

Normally, the input commands are as follows:
```
python run.py --dataset xxx --known_cls_ratio xxx --labeled_ratio xxx --config_file_name xxx
```

Notice that if you want to train the model, save the model, or save the testing results, you need to add related parameters (--train, --save_model, --save_results)

### Results
The detailed results can be seen in [results.md](results/results.md).
#### Overall Performance

| | | BANKING     |  | OOS      |  |  StackOverflow     |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR* | Methods | Accuracy | F1-score | Accuracy  |F1-score  | Accuracy | F1-score | 
|0.25|MSP|42.19|49.92|53.38|51.23|27.91|37.49|
|0.25|SEG|48.73|51.49|52.18|47.0|23.33|34.4|
|0.25|OpenMax|47.76|53.18|70.27|63.03|38.97|45.35|
|0.25|LOF|66.73|63.38|87.77|78.13|25.02|35.29|
|0.25|DOC|70.31|65.74|86.08|75.86|57.75|57.34|
|0.25|DeepUnk|70.68|65.57|87.18|77.32|40.03|45.64|
|0.25|(K+1)-way|76.66|68.44|85.36|74.43|49.75|50.82|
|0.25|MDF|77.17|46.85|76.56|50.34|74.1|53.95|
|0.25|ARPL|76.8|64.01|84.51|73.44|66.76|62.62|
|0.25|ADB|79.33|71.63|88.3|78.23|86.75|79.85|
|0.25|DA-ADB|81.19|73.73|89.48|79.92|89.07|82.83|
|||||||||
|0.5|MSP|61.67|72.51|66.68|72.7|53.23|62.7|
|0.5|SEG|55.11|63.32|60.67|62.55|43.04|55.1|
|0.5|OpenMax|65.53|74.64|80.22|79.86|60.27|67.72|
|0.5|LOF|71.13|76.26|85.22|83.86|44.56|56.57|
|0.5|DOC|74.6|78.24|85.19|83.89|73.88|76.8|
|0.5|DeepUnk|71.01|75.41|84.95|83.35|55.46|64.78|
|0.5|(K+1)-way|74.65|77.83|82.19|81.56|62.57|68.81|
|0.5|MDF|60.18|64.1|60.72|61.61|56.46|61.47|
|0.5|ARPL|74.11|77.77|80.36|80.88|75.65|77.87|
|0.5|ADB|79.61|81.34|86.54|85.16|86.49|85.54|
|0.5|DA-ADB|81.51|82.53|87.93|85.64|87.78|86.91|
|||||||||
|0.75|MSP|77.08|84.33|76.19|83.48|73.2|78.7|
|0.75|SEG|64.65|69.54|42.78|42.7|62.72|69.97|
|0.75|OpenMax|78.32|84.95|75.36|71.17|75.78|80.9|
|0.75|LOF|77.21|83.64|85.07|87.2|65.05|71.87|
|0.75|DOC|78.94|83.79|85.93|87.87|80.55|84.37|
|0.75|DeepUnk|74.73|81.12|84.61|86.53|71.56|77.63|
|0.75|(K+1)-way|79.18|84.71|83.51|86.66|74.0|78.95|
|0.75|MDF|64.59|74.76|63.98|72.02|62.98|71.12|
|0.75|ARPL|79.6|85.16|81.29|86.0|79.64|83.85|
|0.75|ADB|81.39|86.11|86.99|88.94|82.89|86.11|
|0.75|DA-ADB|81.12|85.65|87.39|88.41|83.56|86.84|

*KIR means "Known Intent Ratio".


#### Fine-grained Performance

|  | | BANKING     |  | OOS      |  |  StackOverflow     |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR | Methods | Open | Known | Open | Known | Open | Known | 
|0.25|MSP|39.42|50.47|59.26|51.02|11.66|42.66|
|0.25|SEG|51.58|51.48|59.22|46.67|4.19|40.44|
|0.25|OpenMax|48.52|53.42|77.51|62.65|34.52|47.51|
|0.25|LOF|72.64|62.89|91.96|77.77|7.14|40.92|
|0.25|DOC|76.64|65.16|90.78|75.46|62.5|56.3|
|0.25|DeepUnk|76.98|64.97|91.61|76.95|36.87|47.39|
|0.25|(K+1)-way|82.66|67.7|90.27|74.02|52.23|50.54|
|0.25|MDF|85.7|44.8|84.89|49.43|83.03|48.13|
|0.25|ARPL|83.39|62.99|89.63|73.01|72.95|60.55|
|0.25|ADB|85.05|70.92|92.36|77.85|90.96|77.62|
|0.25|DA-ADB|86.57|73.05|93.2|79.57|92.65|80.87|
|||||||||
|0.5|MSP|46.29|73.2|63.71|72.82|26.94|66.28|
|0.5|SEG|43.03|63.85|61.34|62.57|4.72|60.14|
|0.5|OpenMax|55.03|75.16|82.15|79.83|46.11|69.88|
|0.5|LOF|66.81|76.51|87.57|83.81|5.18|61.71|
|0.5|DOC|72.66|78.38|87.45|83.84|71.18|77.37|
|0.5|DeepUnk|67.8|75.61|87.48|83.3|35.8|67.67|
|0.5|(K+1)-way|72.58|77.97|84.25|81.52|51.69|70.53|
|0.5|MDF|57.72|64.27|62.31|61.6|50.19|62.6|
|0.5|ARPL|71.79|77.93|81.81|80.87|73.97|78.26|
|0.5|ADB|79.43|81.39|88.6|85.12|87.7|85.32|
|0.5|DA-ADB|81.93|82.54|90.1|85.58|88.86|86.71|
|||||||||
|0.75|MSP|46.05|84.99|63.86|83.65|37.86|81.42|
|0.75|SEG|37.22|70.1|40.74|42.72|6.0|74.24|
|0.75|OpenMax|53.02|85.5|75.18|71.14|49.69|82.98|
|0.75|LOF|54.19|84.15|82.81|87.24|5.22|76.31|
|0.75|DOC|63.51|84.14|83.87|87.91|65.32|85.64|
|0.75|DeepUnk|50.57|81.65|82.67|86.57|34.38|80.51|
|0.75|(K+1)-way|59.89|85.14|79.59|86.72|45.22|81.2|
|0.75|MDF|33.43|75.47|51.33|72.21|28.52|73.96|
|0.75|ARPL|61.26|85.58|74.67|86.1|62.99|85.24|
|0.75|ADB|67.34|86.44|84.85|88.97|74.1|86.91|
|0.75|DA-ADB|69.37|85.93|86.0|88.43|74.55|87.66|

“Open” and “Known” denote the macro f1-score over open class and known classes respectively.


## Tutorials
### a. How to add a new dataset? 
1. Prepare Data  
Create a new directory to store your dataset in the [data](../data) directory. You should provide the train.tsv, dev.tsv, and test.tsv, with the same formats as in the provided [datasets](../data/banking).

2. Dataloader Setting  
Calculate the maximum sentence length (token unit) and count the labels of the dataset. Add them in the [file](./configs/__init__.py) as follows:  
```
max_seq_lengths = {
    'new_dataset': max_length
}
benchmark_labels = {
    'new_dataset': label_list
}
```

### b. How to add a new backbone?

1. Add a new backbone in the [backbones](./backbones) directory. For example, we provide some bert-based backbones in the [file](./backbones/bert.py).

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
Create a new file, named "method_name.py" in the [configs](./configs) directory, and set the hyper-parameters for the method (an example can be seen in [MSP.py](./configs/MSP.py)). 

2. Dataloader Setting  
Add the dataloader mapping if you use new backbone for the method. For example, the bert-based model corresponds to the bert dataloader as follows.
```
from .bert_loader import BERT_Loader
backbone_loader_map = {
    'bert': BERT_Loader,
    'bert_xxx': BERT_Loader,
}
```

3. Add Methods  (Take MSP as an example)
- Create a new directory, named "MSP" in the [methods](./methods) directory. 

- Add the manager file for MSP. The file should include the method manager class (e.g., MSPManager), which includes training, evalutation, and testing modules for the method. An example can be seen in [manager.py](./methods/MSP/manager.py).  

- Add the related method dependency in [__init__.py](./methods/__init__.py) as below:
```
from .MSP.manager import xxxManager
method_map = {
    'MSP': MSPManager
}
```
(The key corresponds to the input parameter "method")

4. Run Examples
Add a script in the [examples](./examples) directory, and configure the parsing parameters in the [run.py](./run.py). You can also run the programs serially by setting the combination of different parameters. A running example is shown in [run_MSP.sh](./examples/run_MSP.sh).

## Citation

If this work is helpful, or you want to use the codes and results in this repo, please cite the following papers:

* [TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition](https://aclanthology.org/2021.acl-demo.20/)
* [Learning Discriminative Representations and Decision Boundaries for Open Intent Detection](https://ieeexplore.ieee.org/document/10097558)

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
    year = {2023}, 
    doi = {10.1109/TASLP.2023.3265203} 
 } 
```
