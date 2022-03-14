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
If you are interested in this work, and want to use the codes in this repo, please star/fork this repo and cite the following works:
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
@article{Zhang_Xu_Lin_2021, 
      title={Deep Open Intent Classification with Adaptive Decision Boundary}, 
      volume={35}, 
      number={16}, 
      journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
      author={Zhang, Hanlei and Xu, Hua and Lin, Ting-En}, 
      year={2021}, 
      month={May}, 
      pages={14374-14382} 
}
```



