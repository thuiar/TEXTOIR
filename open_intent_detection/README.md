# Open Intent Detection

This package provides the toolkit for open intent detection implemented with PyTorch.

## Introduction

Open intent detection aims to identify n-class known intents, and detect the one-class open intent, which is regarded as an open classification problem. The following is an example:

<img src="figs/open_intent_detection.png" width="360" height = "200">

We collect benchmark intent datasets, and reproduce related methods to our best. For the convenience of users, we provide flexible and extensible interfaces to add new methods. Welcome to contact us (zhang-hl20@mails.tsinghua.edu.cn) to add your methods!

## Basic Information

### Benchmark Datasets

* [BANKING](https://arxiv.org/pdf/2003.04807.pdf)
* [OOS](https://arxiv.org/pdf/1909.02027.pdf) 
* [StackOverflow](https://aclanthology.org/W15-1509.pdf)

### Integrated Models

* [Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training](https://aclanthology.org/2021.acl-long.273) (MixUp, ACL IJCNLP 2021)
* [Unknown Intent Detection Using Gaussian Mixture Model with an Application to Zero-shot Intent Classification](https://aclanthology.org/2020.acl-main.99) (SEG, ACL 2020)
* [Deep Open Intent Classification with Adaptive Decision Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/17690) (ADB, AAAI 2021)
* [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (DeepUnk, ACL 2019)
* [DOC: Deep Open Classification of Text Documents](https://aclanthology.org/D17-1314.pdf) (DOC, EMNLP 2017)
* [A Baseline For Detecting Misclassified and Out-of-distribution Examples in Neural Networks](https://arxiv.org/pdf/1610.02136.pdf) (MSP, ICLR 2017) 
* [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (OpenMax, CVPR 2016)

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
|0.25|MSP|41.84|50.03|49.78|49.42|27.91|37.49|
|0.25|SEG|49.73|52.03|53.34|47.57|23.35|34.59|
|0.25|OpenMax|47.76|53.18|70.27|63.03|38.97|45.35|
|0.25|LOF|66.73|63.38|87.77|78.13|25.02|35.29|
|0.25|DOC|70.31|65.74|86.08|75.86|57.75|57.34|
|0.25|DeepUnk|70.68|65.57|87.18|77.32|40.03|45.64|
|0.25|(K+1)-way|75.43|68.31|86.98|76.58|53.05|53.12|
|0.25|ADB|79.94|72.08|88.21|78.14|86.7|79.79|
|0.25|DA-ADB|81.09|73.65|89.49|79.95|89.03|82.81|
|0.5|MSP|59.8|71.4|62.71|70.33|53.23|62.7|
|0.5|SEG|54.66|62.86|60.54|62.51|43.04|55.1|
|0.5|OpenMax|65.53|74.64|80.22|79.86|60.27|67.72|
|0.5|LOF|71.13|76.26|85.22|83.86|44.56|56.57|
|0.5|DOC|74.6|78.24|85.19|83.89|73.88|76.8|
|0.5|DeepUnk|71.01|75.41|84.95|83.35|55.46|64.78|
|0.5|(K+1)-way|74.66|78.13|83.71|82.85|63.54|69.26|
|0.5|ADB|79.52|81.33|86.47|85.11|86.51|85.55|
|0.5|DA-ADB|81.64|82.6|87.96|85.64|87.79|86.92|
|0.75|MSP|75.9|83.49|72.86|81.61|73.2|78.7|
|0.75|SEG|64.54|69.37|42.97|42.49|62.63|69.86|
|0.75|OpenMax|78.32|84.95|75.36|71.17|75.78|80.9|
|0.75|LOF|77.21|83.64|85.07|87.2|65.05|71.87|
|0.75|DOC|78.94|83.79|85.93|87.87|80.55|84.37|
|0.75|DeepUnk|74.73|81.12|84.61|86.53|71.56|77.63|
|0.75|(K+1)-way|79.9|85.22|85.31|87.9|74.72|79.47|
|0.75|ADB|81.35|86.08|86.98|88.95|82.84|86.07|
|0.75|DA-ADB|81.18|85.68|87.46|88.47|83.63|86.89|

*KIR means "Known Intent Ratio".


#### Fine-grained Performance

|  | | BANKING     |  | OOS      |  |  StackOverflow     |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR | Methods | Open | Known | Open | Known | Open | Known | 
|0.25|MSP|38.84|50.62|54.74|49.28|11.66|42.66|
|0.25|SEG|52.97|51.98|60.59|47.23|4.36|40.63|
|0.25|OpenMax|48.52|53.42|77.51|62.65|34.52|47.51|
|0.25|LOF|72.64|62.89|91.96|77.77|7.14|40.92|
|0.25|DOC|76.64|65.16|90.78|75.46|62.5|56.3|
|0.25|DeepUnk|76.98|64.97|91.61|76.95|36.87|47.39|
|0.25|(K+1)-way|81.52|67.61|91.44|76.19|56.31|52.48|
|0.25|ADB|85.57|71.37|92.3|77.77|90.91|77.56|
|0.25|DA-ADB|86.49|72.97|93.2|79.6|92.61|80.84|
|0.5|MSP|42.13|72.17|57.49|70.5|26.94|66.28|
|0.5|SEG|42.35|63.4|61.13|62.52|4.72|60.14|
|0.5|OpenMax|55.03|75.16|82.15|79.83|46.11|69.88|
|0.5|LOF|66.81|76.51|87.57|83.81|5.18|61.71|
|0.5|DOC|72.66|78.38|87.45|83.84|71.18|77.37|
|0.5|DeepUnk|67.8|75.61|87.48|83.3|35.8|67.67|
|0.5|(K+1)-way|72.38|78.29|85.84|82.82|53.68|70.81|
|0.5|ADB|79.32|81.38|88.54|85.06|87.72|85.33|
|0.5|DA-ADB|82.1|82.61|90.14|85.58|88.86|86.72|
|0.75|MSP|41.64|84.21|56.26|81.83|37.86|81.42|
|0.75|SEG|37.58|69.92|41.6|42.5|6.38|74.09|
|0.75|OpenMax|53.02|85.5|75.18|71.14|49.69|82.98|
|0.75|LOF|54.19|84.15|82.81|87.24|5.22|76.31|
|0.75|DOC|63.51|84.14|83.87|87.91|65.32|85.64|
|0.75|DeepUnk|50.57|81.65|82.67|86.57|34.38|80.51|
|0.75|(K+1)-way|62.13|85.62|82.39|87.95|47.57|81.6|
|0.75|ADB|67.32|86.4|84.81|88.99|74.02|86.87|
|0.75|DA-ADB|69.51|85.96|86.09|88.49|74.66|87.71|
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
@article{zhang2022towards,
  title={Towards Open Intent Detection},
  author={Zhang, Hanlei and Xu, Hua and Zhao, Shaojie and Zhou, Qianrui},
  journal={arXiv preprint arXiv:2203.05823},
  year={2022}
}
```
