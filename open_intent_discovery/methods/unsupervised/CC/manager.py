import logging
import numpy as np
import torch
import torch.nn as nn
from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from utils.functions import save_model
from losses import contrastive_loss

class CCmanager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.device = model.device
        self.num_labels = data.num_labels
        
        loader = data.dataloader
        self.train_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.test_outputs['loader']
        self.train_input_ids, self.train_input_mask, self.train_segment_ids = \
            loader.train_outputs['input_ids'], loader.train_outputs['input_mask'], loader.train_outputs['segment_ids']
        self.augdataloader = self.get_augment_dataloader(args)
        
        self.set_model_optimizer(args, data, model)
        
        self.instance_temperature = 0.7 
        self.cluster_temperature = 1.0
        self.criterion_instance = contrastive_loss.InstanceLoss(args.train_batch_size, self.instance_temperature, self.device) 
        self.criterion_cluster = contrastive_loss.ClusterLoss(self.num_labels, self.cluster_temperature, self.device) 
    
    def set_model_optimizer(self, args, data, model):
            
        self.model = model.set_model(args, data, 'bert', args.freeze_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device

    def batch_chunk(self, x):
        x1, x2 = torch.chunk(input=x, chunks=2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        return x1, x2

    def train(self, args, data):
         
        self.logger.info('CC training starts...')
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            tr_loss, nb_tr_steps = 0, 0
            self.model.train()
            for batch in tqdm(self.augdataloader, desc="Training(All)"):
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                                               
                with torch.set_grad_enabled(True):
                    
                    input_ids_a,  input_ids_b = self.batch_chunk(input_ids)
                    input_mask_a,  input_mask_b = self.batch_chunk(input_mask)
                    segment_ids_a,  segment_ids_b = self.batch_chunk(segment_ids)
                    
                    x_i = self.model(input_ids_a, segment_ids_a, input_mask_a)               
                    x_j = self.model(input_ids_b, segment_ids_b, input_mask_b)
         
                    z_i, z_j, c_i, c_j = self.model.get_features(x_i, x_j)
                    loss_instance = self.criterion_instance(z_i, z_j)
                    loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = loss_instance + loss_cluster              

                    self.optimizer.zero_grad()
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_steps += 1
                                
                    self.optimizer.step()
                    self.scheduler.step()
                
            train_loss = tr_loss / nb_tr_steps
            
            self.logger.info("***** Epoch: %s: train results *****", str(epoch))
            self.logger.info("  train_loss = %s",  str(train_loss))
        
        self.logger.info('CC training finished...')
        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):

        feats, y_true = self.get_outputs(args, mode = 'test')
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
    
    def get_outputs(self, args, mode):
        
        if mode == 'test':
            dataloader = self.test_dataloader
            
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
         
            with torch.set_grad_enabled(False):
                pooled_output = self.model(input_ids, segment_ids, input_mask)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
 
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        return feats, y_true

    def get_augment_dataloader(self, args):

        train_input_ids = self.train_input_ids.unsqueeze(1)
        train_input_mask = self.train_input_mask.unsqueeze(1)
        train_segment_ids = self.train_segment_ids.unsqueeze(1)

        train_input_ids = torch.cat(([train_input_ids, train_input_ids]), dim = 1)
        train_input_mask = torch.cat(([train_input_mask, train_input_mask]), dim = 1)
        train_segment_ids = torch.cat(([train_segment_ids, train_segment_ids]), dim = 1)

        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler,  batch_size = args.train_batch_size, drop_last=True)

        return train_dataloader

