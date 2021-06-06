## Introduction
This is a toolkit for text open intent detection.

### Usage
### Parameters
config_file_name: xxx / xxx.py, containing the parameters (you can add your own file).
dataset: banking | oos | stackoverflow | snips
known_cls_ratio: 0.25 | 0.5 | 0.75 (default) | 1.0  
labeled_ratio: 0.2 | 0.4 | 0.6 | 0.8 | 1.0 (default)  
method: ADB (default) | DeepUnk | DOC | MSP | OpenMax
train: if this parameter is not set, you need to provide / pre-train the model in advance.
save_model: if this parameter is not set, the trained model is not saved. 
save_results: if this parameter is not set, the results is not saved. 
backbone: bert (default) | bert_deepunk
seed: 0 (default) 

### An Example

#### Train 
Full command:  
python run.py --num_train_epochs 1 --method ADB --known_cls_ratio 0.25 --backbone bert_deepunk --labeled_ratio 0.1 --config_file_name ADB_LMCosine --seed 0 --train  --save_model --save_results