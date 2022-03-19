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

||Dataset||BANKING||||OOS||||StackOverflow|||
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KIR*|Methods|Known|Open|F1|Acc|Known|Open|F1|Acc|Known|Open|F1|Acc|
|0.25|MSP|50.62|38.84|50.03|41.84|49.28|54.74|49.42|49.78|42.66|11.66|37.49|27.91|
|0.25|SEG|51.98|52.97|52.03|49.73|47.23|60.59|47.57|53.34|40.63|4.36|34.59|23.35|
|0.25|OpenMax|53.42|48.52|53.18|47.76|62.65|77.51|63.03|70.27|47.51|34.52|45.35|38.97|
|0.25|LOF|62.89|72.64|63.38|66.73|77.77|91.96|78.13|87.77|40.92|7.14|35.29|25.02|
|0.25|DOC|65.16|76.64|65.74|70.31|75.46|90.78|75.86|86.08|56.3|62.5|57.34|57.75|
|0.25|DeepUnk|64.97|76.98|65.57|70.68|76.95|91.61|77.32|87.18|47.39|36.87|45.64|40.03|
|0.25|(K+1)-way|67.61|81.52|68.31|75.43|76.19|91.44|76.58|86.98|52.48|56.31|53.12|53.05|
|0.25|ADB|71.37|85.57|72.08|79.94|77.77|92.3|78.14|88.21|77.56|90.91|79.79|86.7|
|0.25|DA-ADB|72.97|86.49|73.65|81.09|79.6|93.2|79.95|89.49|80.84|92.61|82.81|89.03|
|0.5|MSP|72.17|42.13|71.4|59.8|70.5|57.49|70.33|62.71|66.28|26.94|62.7|53.23|
|0.5|SEG|63.4|42.35|62.86|54.66|62.52|61.13|62.51|60.54|60.14|4.72|55.1|43.04|
|0.5|OpenMax|75.16|55.03|74.64|65.53|79.83|82.15|79.86|80.22|69.88|46.11|67.72|60.27|
|0.5|LOF|76.51|66.81|76.26|71.13|83.81|87.57|83.86|85.22|61.71|5.18|56.57|44.56|
|0.5|DOC|78.38|72.66|78.24|74.6|83.84|87.45|83.89|85.19|77.37|71.18|76.8|73.88|
|0.5|DeepUnk|75.61|67.8|75.41|71.01|83.3|87.48|83.35|84.95|67.67|35.8|64.78|55.46|
|0.5|(K+1)-way|78.29|72.38|78.13|74.66|82.82|85.84|82.85|83.71|70.81|53.68|69.26|63.54|
|0.5|ADB|81.38|79.32|81.33|79.52|85.06|88.54|85.11|86.47|85.33|87.72|85.55|86.51|
|0.5|DA-ADB|82.61|82.1|82.6|81.64|85.58|90.14|85.64|87.96|86.72|88.86|86.92|87.79|
|0.75|MSP|84.21|41.64|83.49|75.9|81.83|56.26|81.61|72.86|81.42|37.86|78.7|73.2|
|0.75|SEG|69.92|37.58|69.37|64.54|42.5|41.6|42.49|42.97|74.09|6.38|69.86|62.63|
|0.75|OpenMax|85.5|53.02|84.95|78.32|71.14|75.18|71.17|75.36|82.98|49.69|80.9|75.78|
|0.75|LOF|84.15|54.19|83.64|77.21|87.24|82.81|87.2|85.07|76.31|5.22|71.87|65.05|
|0.75|DOC|84.14|63.51|83.79|78.94|87.91|83.87|87.87|85.93|85.64|65.32|84.37|80.55|
|0.75|DeepUnk|81.65|50.57|81.12|74.73|86.57|82.67|86.53|84.61|80.51|34.38|77.63|71.56|
|0.75|(K+1)-way|85.62|62.13|85.22|79.9|87.95|82.39|87.9|85.31|81.6|47.57|79.47|74.72|
|0.75|ADB|86.4|67.32|86.08|81.35|88.99|84.81|88.95|86.98|86.87|74.02|86.07|82.84|
|0.75|DA-ADB|85.96|69.51|85.68|81.18|88.49|86.09|88.47|87.46|87.71|74.66|86.89|83.63|

*KIR means "Known Intent Ratio".
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
