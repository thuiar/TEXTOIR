import torch
import torch.nn.functional as F
import logging
import os
import torch.nn as nn
import numpy as np
import copy

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, MemoryBank, fill_memory_bank, view_generator, set_seed
from utils.neighbor_dataset import NeighborsDataset
from torch.utils.data import DataLoader
from .pretrain import PretrainMTP_CLNNManager
from utils.metrics import clustering_score
from transformers import AutoTokenizer

class MTP_CLNNManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        pretrain_manager = PretrainMTP_CLNNManager(args, data, model) 
        
        set_seed(args.seed)
        self.logger = logging.getLogger(logger_name)
        
        loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader']
        self.train_dataset = loader.train_outputs['semi_data']

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_model) 
        self.generator = view_generator(self.tokenizer, args)
        self.temp=0.07
        
        if args.pretrain:

            self.pretrained_model = pretrain_manager.model
            self.set_model_optimizer(args, data, model)
            self.num_labels = data.num_labels 
            self.load_pretrained_model(self.pretrained_model)
        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.set_model_optimizer(args, data, model)
            self.num_labels = data.num_labels 
            self.model = restore_model(self.model, args.model_output_dir)

        topk = {'banking': 50, 'clinc': 60, 'stackoverflow': 300}
        
        if args.cluster_num_factor > 1:

            self.logger.info('num_labels is %s, Length of train_dataset is %s', str(self.num_labels), str(len(self.train_dataset)))
            args.topk = int((len(self.train_dataset) * 0.5) / self.num_labels)
        else:
            args.topk = topk[args.dataset]

        self.logger.info('Topk for %s is %s', str(args.dataset), str(args.topk))
        
    def set_model_optimizer(self, args, data, model):
        
        if args.dataset == 'stackoverflow':
            args.lr = 1e-6
            
        args.backbone = 'bert_MTP'
        self.model = model.set_model(args, data, 'bert', args.freeze_train_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_examples), args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device 
        self.criterion = self.model.loss_cl           
        
    def train(self, args, data): 
        
        indices = self.get_neighbor_inds(args, data)
        self.get_neighbor_dataset(args, data, indices)
        best_eval_score = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
        
            for batch in tqdm(self.train_dataloader_neighbor, desc="Iteration"):

                anchor = tuple(t.to(self.device) for t in batch["anchor"]) 
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) 
                pos_neighbors = batch["possible_neighbors"]
                data_inds = batch["index"] 

                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)
                X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
            
                with torch.set_grad_enabled(True):
                    f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                    loss = self.criterion(f_pos, mask=adjacency, temperature=self.temp, device = self.device)
                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1
                    
            loss = tr_loss / nb_tr_steps

            self.logger.info("***** Epoch: %s *****", str(epoch))
            self.logger.info('Training Loss: %f', np.round(loss, 5))
                    
            if ((epoch + 1) % args.update_per_epoch) == 0:
                self.logger.info("Update neighbors...")
                
                indices = self.get_neighbor_inds(args, data)
                self.get_neighbor_dataset(args, data, indices)
            
        if args.save_model:
            save_model(self.model, args.model_output_dir)
            
    def test(self, args, data):
        
        feats, y_true = self.get_outputs(args, mode = 'test', model = self.model, get_feats = True)
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        y_pred = km.labels_
        
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def get_outputs(self, args, mode, model, get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
         
            with torch.set_grad_enabled(False):
                pooled_output = model(X)["hidden_states"] 
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:  
            feats = total_features.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return feats, y_true
        
    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        self.model.load_state_dict(pretrained_dict, strict=False)

    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(self.train_dataset, indices)
        self.train_dataloader_neighbor = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    def get_neighbor_inds(self, args, data):
        """get indices of neighbors"""
        memory_bank = MemoryBank(len(self.train_dataset), args.feat_dim, self.num_labels, 0.1)
        fill_memory_bank(self, self.train_dataloader, self.model, memory_bank)
        indices = memory_bank.mine_nearest_neighbors(args.topk, args.gpu_id ,calculate_accuracy=False)
        return indices
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 
                if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                    adj[b1][b2] = 1 
                 
        return adj

