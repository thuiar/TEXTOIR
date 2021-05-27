## Introduction
This is a toolkit for text open intent detection.

### Usage
### Parameters
dataset: banking | oos | stackoverflow
known_cls_ratio: 0.25 | 0.5 | 0.75 (default) | 1.0  
labeled_ratio: 0.2 | 0.4 | 0.6 | 0.8 | 1.0 (default)  
method: ADB (default) | DeepUnk | DOC | MSP | OpenMax
train: if this parameter is not set, you need to provide / pre-train the model in advance.
save: if this parameter is not set, the trained model is not saved. 
backbone: bert (default)
seed: 0 (default) 
### An Example
#### Train
python run_detect.py --dataset banking --method ADB --known_cls_ratio 0.25 --labeled_ratio 0.1 --train --save
#### Test
python run_detect.py --dataset banking --method ADB --known_cls_ratio 0.25 --labeled_ratio 0.1
#### Test mode
python run_detect.py --method ADB --train --save

aoligei!!!