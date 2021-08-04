# Open Intent Discovery
This package provides the toolkit for open intent discovery implemented with PyTorch (Semi-supervised) and Tensorflow (Unsupervised).

## Introduction
Open intent discovery aims to identify n-class known intents, and detect one-class open intent. We collect benchmark intent datasets, and reproduce related methods to our best. For the convenience of users, we provide flexible and extensible interfaces to add new methods. Welcome to contact us to add your methods!

Open Intent Detection:  
![Example](figs/open_intent_detection.png =100x "Example")

## Benchmark Datasets
* [BANKING](https://arxiv.org/pdf/2003.04807.pdf)
* [OOS](https://arxiv.org/pdf/1909.02027.pdf) 
* [StackOverflow](https://aclanthology.org/W15-1509.pdf)

## Models

* [Deep Open Intent Classification with Adaptive Decision Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/17690) (ADB, AAAI 2021)
* [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (DeepUnk, ACL 2019)
* [DOC: Deep Open Classification of Text Documents](https://aclanthology.org/D17-1314.pdf) (DOC, EMNLP 2017)
* [A Baseline For Detecting Misclassified and Out-of-distribution Examples in Neural Networks](https://arxiv.org/pdf/1610.02136.pdf) (MSP, ICLR 2017) 
* [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (OpenMax, CVPR 2016)

We welcome any issues and requests for model implementation and bug fix. 

## Experimental Settings
Each dataset is split to training, development, and testing sets. We select partial intents as known (the labeled ratio can be changed) for training, and use all intents for testing. All the unknown intents are regarded as one open class (with token \<UNK> or \<OOS> in our codes).


## Configurations
The basic parameters include parsing parameters about selected dataset, method, setting, etc. More details can be seen in [run.py](./run.py). For specific parameters of each method, we support add configuration files with different hyper-parameters in the [configs](./configs) directory. 

An example can be seen in [configs/ADB.py](./configs/ADB.py). Notice that the config file name is corresponding to the parsing parameter.

Normally, the input commands are as follows:
```
python run.py --dataset xxx --known_cls_ratio xxx --labeled_ratio xxx --config_file_name xxx --train --save_model --save_results
```
Notice that if you want to train the model, save the model, or save the testing results, you need to add related parameters (--train, --save_model, --save_results)

## How to add your own method? (Take MSP as an example)

1. Configs Setting   
1.1 Create a new file, named "MSP.py" in the [configs](./configs) directory, and set the hyper-parameters for the method (an example can be seen in [MSP.py](./configs/MSP.py)).  
1.2 set the [configs/base.py](./configs/base.py) configs as follows:
 
```
from xxx import xxx
from .MSP import MSP_Param

param_map = {
    'xxx': xxx, 'MSP': MSP_Param
}
```
2. Add Methods  
2.1 Create a new directory, named "MSP" in the [methods](./methods) directory.  
2.2 Add the manager file for MSP. The file should include the method manager class (e.g., MSPManager), which includes training, evalutation, and testing modules for the method. An example can be seen in [methods/MSP/manager.py](./methods/MSP/manager.py).  
2.3 Add the related method dependency in [methods/__init__.py](./methods/__init__.py) as below:
```
from xxx import xxx
from .MSP.manager import MSPManager

method_map = {'xxx': xxx, 'MSP': MSPManager}
```

3. Training and Testing  
3.1 Common Parameters (More Details in [this](./configs/base.py) file)    
    --dataset  
    The name of the chosen dataset. Type: str. Supported datasets can be seen in 

    --known_cls_ratio (The class ratio of known intents, type: float)  
    --train (whether to train the model)  
    --save_model (whether to save the well-trained model)  
3.2 An Example:  
    python run.py --dataset banking --known_cls_ratio 0.25 --train --save_model 
    



