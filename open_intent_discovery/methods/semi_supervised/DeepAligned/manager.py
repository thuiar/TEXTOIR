import torch
import torch.nn.functional as F
import numpy as np
import copy
import logging
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment
from losses import loss_map
from utils.functions import save_model, restore_model
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from utils.metrics import clustering_score
from .pretrain import PretrainDeepAlignedManager

class DeepAlignedManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        self.model = model.model
        self.optimizer = model.optimizer
        self.device = model.device

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.dataloader.train_loader, data.dataloader.eval_loader, data.dataloader.test_loader
        self.train_input_ids, self.train_input_mask, self.train_segment_ids = \
            data.dataloader.train_input_ids, data.dataloader.train_input_mask, data.dataloader.train_segment_ids

        self.loss_fct = loss_map[args.loss_fct]
        
        pretrain_manager = PretrainDeepAlignedManager(args, data, model)  
        

        if args.train:

            self.logger.info('Pre-raining start...')
            pretrain_manager.train(args, data)
            self.logger.info('Pre-training finished...')

            self.centroids = None
            self.pretrained_model = pretrain_manager.model

            if args.cluster_num_factor > 1:
                self.num_labels = data.num_labels
                self.num_labels = self.predict_k(args, data) 
            else:
                self.num_labels = data.num_labels 

            self.load_pretrained_model(self.pretrained_model)

        else:

            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            
            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data) 
            else:
                self.num_labels = data.num_labels 

            self.model = restore_model(self.model, args.model_output_dir)

    def train(self, args, data): 

        best_model = None
        wait = 0
        best_eval_score = 0 

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            feats, _ = self.get_outputs(args, mode = 'train', model = self.model, get_feats = True)
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            eval_score = silhouette_score(feats, km.labels_)

            if epoch > 0:
                
                eval_results = {
                    'train_loss': tr_loss,
                    'cluster_silhouette_score': eval_score,
                    'best_cluster_silhouette_score': best_eval_score,   
                }

                self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
                for key in sorted(eval_results.keys()):
                    self.logger.info("  %s = %s", key, str(eval_results[key]))

            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score
            elif eval_score > 0:
                wait += 1
                if wait >= args.wait_patient:
                    break 
            
            pseudo_labels = self.alignment(km, args)
            pseudo_train_dataloader = self.update_pseudo_labels(pseudo_labels, args)
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(pseudo_train_dataloader, desc="Training(All)"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, loss_fct = self.loss_fct, mode = "train")
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
        
        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):
        
        feats, y_true = self.get_outputs(args, mode = 'test', model = self.model, get_feats = True)
        km = KMeans(n_clusters = self.num_labels).fit(feats)
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
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = model(input_ids, segment_ids, input_mask)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                if not get_feats:  
                    total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return feats, y_true

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def predict_k(self, args, data):

        self.logger.info('Predict number of clusters start...')

        feats, _ = self.get_outputs(args, mode = 'train', model = self.pretrained_model, get_feats = True) 

        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        K = len(pred_label_list) - cnt
        
        self.logger.info('Predict number of clusters finish...')
        outputs = {'K': K, 'mean_cluster_size': drop_out}
        for key in outputs.keys():
            self.logger.info("  %s = %s", key, str(outputs[key]))

        return K

    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args):
        train_data = TensorDataset(self.train_input_ids, self.train_input_mask, self.train_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader


            


    
