# TEXTOIR
An Integrated and Extensible Platform for Text Open Intent Recognition.

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
    
