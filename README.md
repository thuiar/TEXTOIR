# TEXTOIR
TEXTOIR is an Integrated and Extensible Platform for Text Open Intent Recognition.

## Introduction
TEXTOIR contains two tasks, which are defined as open intent detection and open intent discovery. Open intent detection aims to identify n-class known intents, and detect one-class open intent. Open intent discovery aims to leverage limited prior knowledge of known intents to find fine-grained known and open intent-wise clusters.

## Benmark Datasets
* [BANKING](https://arxiv.org/pdf/2003.04807.pdf)
* [OOS / CLINC150 (without OOD samples)](https://arxiv.org/pdf/1909.02027.pdf) 
* [StackOverflow](https://aclanthology.org/W15-1509.pdf)

## Integrated Models
### Open Intent Detection

* [Deep Open Intent Classification with Adaptive Decision Boundary](https://arxiv.org/pdf/2012.10209.pdf) (ADB)
* [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (DeepUnk)
* [DOC: Deep Open Classification of Text Documents](https://aclanthology.org/D17-1314.pdf) (DOC)
* [Maximum Softmax Probability](https://arxiv.org/pdf/1610.02136.pdf) (MSP) 
* [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (OpenMax)


### Open Intent Discovery

* Unsupervised Clustering Methods

## Environments

PyTorch  (Cuda version 11.2)  
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge
### open intent detection
pip install pytorch_pretrained_bert, matplotlib, sklearn
### open intent discovery
pip install nltk, gensim, seaborn, tensorflow-gpu, keras, wordcloud, keybert

## Supported Components
### Methods    
#### Open Intent Detection
ADB [x]:  [paper]()  [code]()  
DeepUnk [x]: [paper]()  [code]()                   
DOC [x]:  [paper]()  [code]()    
MSP [x]:   [paper]()  [code]()  
OpenMax [x]:   [paper]()  [code]()  
### Datasets
BANKING  
OOS  
CLINC (OOS without ood samples)  
StackOverflow  
SNIPS  
ATIS  

#### Open Intent Discovery


## Tutorials
### How to add a new method? (Take MSP as an example)

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
    
