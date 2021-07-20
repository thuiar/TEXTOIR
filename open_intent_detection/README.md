# Open Intent Detection
This package provides the toolkit for open intent detection implemented with PyTorch.

## Introduction
Open intent detection aims to identify n-class known intents, and detect one-class open intent. We collect benchmark intent datasets, and reproduce related methods to our best. For the convenience of users, we provide flexible and extensible interfaces to add new methods. Welcome to contact us to add your methods!

## Datasets
* [BANKING](https://arxiv.org/pdf/2003.04807.pdf)
* [OOS](https://arxiv.org/pdf/1909.02027.pdf) 
* [StackOverflow](https://aclanthology.org/W15-1509.pdf)

## Models

* [Maximum Softmax Probability](https://arxiv.org/pdf/1610.02136.pdf) (MSP) 
* [DOC: Deep Open Classification of Text Documents](https://aclanthology.org/D17-1314.pdf) (DOC)
* [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (OpenMax)
* [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (DeepUnk)
* [Deep Open Intent Classification with Adaptive Decision Boundary](https://arxiv.org/pdf/2012.10209.pdf) (ADB)

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

## Usage
### Installation
1. Install PyTorch (Cuda version 11.2)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge  
```
2. Clone the TEXTOIR repository.
```
git clone https://github.com/TEXTOIR
cd TEXTOIR
```
3. Install related environmental dependencies
```
pip install -r requirements.txt
```
4. Quick Start
```
python run.py
```
or
```
sh scripts/run_ADB.sh
```




