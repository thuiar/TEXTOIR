import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME

def mask_tokens(inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix[torch.where(inputs==0)] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels



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
    
    if hasattr(save_model, 'config'):
        with open(model_config_file, "w") as f:
            f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(output_model_file), strict=False)
    return model

def save_results(args, test_results):

    pred_labels_path = os.path.join(args.method_output_dir, 'y_pred.npy')
    np.save(pred_labels_path, test_results['y_pred'])
    true_labels_path = os.path.join(args.method_output_dir, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    del test_results['y_pred']
    del test_results['y_true']

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    import datetime
    created_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    var = [args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.loss_fct, args.seed, args.num_train_epochs, created_time]
    names = ['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'loss', 'seed', 'train_epochs', 'created_time']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(args.result_dir, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

def class_count(labels):
    class_data_num = []
    for l in np.unique(labels):
        num = len(labels[labels == l])
        class_data_num.append(num)
    return class_data_num

def centroids_cal(model, args, data, train_dataloader, device):
    
    model.eval()
    centroids = torch.zeros(data.num_labels, args.feat_dim).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)

    with torch.set_grad_enabled(False):

        for batch in tqdm(train_dataloader, desc="Calculate centroids"):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features = model(input_ids, segment_ids, input_mask, feature_ext=True)
            total_labels = torch.cat((total_labels, label_ids))

            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]
            
    total_labels = total_labels.cpu().numpy()
    centroids /= torch.tensor(class_count(total_labels)).float().unsqueeze(1).to(device)
    
    return centroids

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
