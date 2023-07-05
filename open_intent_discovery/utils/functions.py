import os
import torch
import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F
import tensorflow as tf
from tqdm import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' 
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
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
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results, debug_args = None):
    
    if 'y_pred' in test_results.keys():
        pred_labels_path = os.path.join(args.method_output_dir, 'y_pred.npy')

        del test_results['y_pred']

    if 'y_true' in test_results.keys():
        true_labels_path = os.path.join(args.method_output_dir, 'y_true.npy')

        del test_results['y_true']

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    var = [args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.logger_file_name, args.seed]
    names = ['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'logger_file_name', 'seed']
    
    if debug_args is not None:
        var.extend([args[key] for key in debug_args.keys()])
        names.extend(debug_args.keys())
    
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
        df1 = df1.append(new,ignore_index=True)
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
    centroids = torch.zeros(args.num_labels, args.feat_dim).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    total_features = torch.empty((0,args.feat_dim)).to(device)
    
    with torch.set_grad_enabled(False):

        for batch in tqdm(train_dataloader, desc="Calculate centroids"):
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            features, logits = model(input_ids, segment_ids, input_mask, feature_ext=True)
            
            total_labels = torch.cat((total_labels, label_ids))
            total_features = torch.cat((total_features, features))
            
            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]
    
    y_true = total_labels.cpu().numpy()

    centroids /= torch.tensor(class_count(y_true)).float().unsqueeze(1).to(device)
    
    return centroids, total_features, total_labels

def plot_confusion_matrix(cm, classes, save_name, normalize=False, title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues, save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.switch_backend('agg')
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        plt.savefig(save_name)

def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
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

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):

        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, gpu_id, calculate_accuracy=True):
        
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1] 
        index = faiss.IndexFlatIP(dim)
        
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) 
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()    #min -1
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) 
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

@torch.no_grad()
def fill_memory_bank(self, loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):

        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch   #min 0 
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        feature = model(X)["hidden_states"] 

        memory_bank.update(feature, label_ids)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
    
class view_generator:
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=self.args.rtr_prob)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos
    
    def random_token_erase(self, input_ids, input_mask):
        
        aug_input_ids = []
        aug_input_mask = []
        
        for inp_i, inp_m in zip(input_ids, input_mask):
            
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_i, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            masked_inds = np.random.choice(inds, size = int(len(inds) * self.args.re_prob), replace = False)
            sent_masked_inds = sent_tokens_inds[masked_inds]

            inp_i = np.delete(inp_i, sent_masked_inds)
            inp_i = F.pad(inp_i, (0, self.args.max_seq_length - len(inp_i)), 'constant', 0)

            inp_m = np.delete(inp_m, sent_masked_inds)
            inp_m = F.pad(inp_m, (0, self.args.max_seq_length - len(inp_m)), 'constant', 0)
    
            aug_input_ids.append(inp_i)
            aug_input_mask.append(inp_m)
        
        aug_input_ids = torch.stack(aug_input_ids, dim=0)
        aug_input_mask = torch.stack(aug_input_mask, dim=0)
        
        return aug_input_ids, aug_input_mask