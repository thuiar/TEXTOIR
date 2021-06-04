
import os
import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME


class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"
    
def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results):
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
    names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(args.result_dir, args.result_file_name)
    
    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

# def debug(outputs, data, manager, args):

#     # args_attrs = ["max_seq_length","feat_dim","warmup_proportion","freeze_bert_parameters","task_name",
#     #               "known_cls_ratio","labeled_ratio","method","seed","gpu_id","num_train_epochs","lr",
#     #               "train_batch_size","eval_batch_size","wait_patient","threshold"]

#     print('-----------------Data--------------------')
#     data_attrs = ["data_dir","n_known_cls","num_labels","all_label_list","known_label_list"]

#     for attr in data_attrs:
#         attr_name = attr
#         attr_value = data.__getattribute__(attr)
#         print(attr_name,':',attr_value)

#     print('-----------------Args--------------------')
#     for k in list(vars(args).keys()):
#         print(k,':',vars(args)[k])

#     print('-----------------Manager--------------------')
#     if args.train:
#         manager_attrs = ["device","best_eval_score","test_results"]
#     else:
#         manager_attrs = ["device", "test_results"]

#     for attr in manager_attrs:
#         attr_name = attr
#         attr_value = manager.__getattribute__(attr)
#         print(attr_name,':',attr_value)
    
#     y_true, y_pred = outputs[0], outputs[1]
#     predictions = list([data.label_list[idx] for idx in y_pred]) 
#     true_labels = list([data.label_list[idx] for idx in y_true]) 
#     print('-----------------Prediction Example--------------------')
#     show_num = 10
#     for i,example in enumerate(data.test_examples):
#         if i >= show_num:
#             break
#         sentence = example.text_a
#         true_label = true_labels[i]
#         predict_label = predictions[i]
#         print(i,':',sentence)
#         print('Pred: {}; True: {}'.format(predict_label, true_label))