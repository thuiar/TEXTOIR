from turtle import distance
import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
import time

from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm
from itertools import cycle
from losses import loss_map
from utils.functions import save_model, restore_model
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

class PretrainUSNIDManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        args.num_labels = data.n_known_cls
        self.n_known_cls = data.n_known_cls
        
        self.set_model_optimizer(args, data, model)
        
        self.loader = data.dataloader
        self.train_outputs = self.loader.train_outputs
        self.train_unlabeled_outputs = self.loader.train_unlabeled_outputs
        self.train_labeled_outputs = self.loader.train_labeled_outputs
        self.train_labeled_dataloader = self.loader.train_labeled_outputs['loader']
        self.train_dataloader = self.loader.train_outputs['loader']
        self.eval_dataloader = self.loader.eval_outputs['loader']
        self.test_dataloader = self.loader.test_outputs['loader']        

        self.criterion = loss_map['CrossEntropyLoss']
        self.contrast_criterion = loss_map['SupConLoss']
 
        if args.pretrain:
            
            self.logger.info('Pre-raining start...')
            self.train(args, data)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.method_output_dir, 'pretrain'))
            
        if args.cluster_num_factor > 1:
            self.num_labels = data.num_labels
            self.num_labels = self.predict_k(args, data) 

        self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, data, 'bert', args.freeze_pretrain_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_unlabeled_examples), args.pretrain_batch_size, \
            args.num_train_epochs, args.lr_pre, args.warmup_proportion)
        
        self.device = model.device
        
    def batch_chunk(self, x):
        x1, x2 = torch.chunk(input=x, chunks=2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        return x1, x2
       
    def train(self, args, data):
        
        wait = 0
        best_model = None
        best_eval_score = 0
        
        train_unlabeled_data = self.train_unlabeled_outputs['data']
        
        contrast_dataloader = DataLoader(train_unlabeled_data, shuffle = True, batch_size = args.pretrain_batch_size)
        
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
               
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0  
            
            for step, (batch_labeled, batch_unlabeled) in enumerate(tqdm(zip(cycle(self.train_labeled_dataloader), contrast_dataloader), desc = "Iteration")):
                
                batch_labeled = tuple(t.to(self.device) for t in batch_labeled)
                labeled_input_ids, labeled_input_mask, labeled_segment_ids, labeled_label_ids = batch_labeled
                batch_unlabeled = tuple(t.to(self.device) for t in batch_unlabeled)
                unlabeled_input_ids, unlabeled_input_mask, unlabeled_segment_ids, unlabeled_label_ids = batch_unlabeled
                                
                input_ids = torch.cat((labeled_input_ids, unlabeled_input_ids))
                input_mask = torch.cat((labeled_input_mask, unlabeled_input_mask))
                segment_ids = torch.cat((labeled_segment_ids, unlabeled_segment_ids))
                label_ids = torch.cat((labeled_label_ids, unlabeled_label_ids))
                
                with torch.set_grad_enabled(True):
                    
                    labeled_mlp_output, labeled_logits = self.model(labeled_input_ids, labeled_segment_ids, labeled_input_mask)
                    loss_ce_labeled = self.criterion(labeled_logits, labeled_label_ids)
                    
                    aug_mlp_output_a, logits_a = self.model(input_ids, segment_ids, input_mask)
                    aug_mlp_output_b, logits_b = self.model(input_ids, segment_ids, input_mask)
                
                    batch_size = logits_a.shape[0]
                    
                    norm_logits = F.normalize(aug_mlp_output_a)
                    norm_aug_logits = F.normalize(aug_mlp_output_b)
                    
                    labels_expand = label_ids.expand(batch_size, batch_size)
                    mask = torch.eq(labels_expand, labels_expand.T).long()
                    mask[label_ids == -1, :] = 0
                    
                    
                    logits_mask = torch.scatter(
                        mask,
                        0,
                        torch.arange(batch_size).unsqueeze(0).to(self.device),
                        1
                    )     
                        
                    contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    loss_contrast = self.contrast_criterion(contrastive_logits, mask = logits_mask, temperature = args.pretrain_temperature, device = self.device)
                    
                    loss = loss_contrast + loss_ce_labeled
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                   
            loss = tr_loss / nb_tr_steps
            
            outputs = self.get_outputs(args, mode = 'eval')
            
            y_true = outputs['y_true']
            y_pred = outputs['y_pred']
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            
            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score': best_eval_score,
            }
            
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
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
               
        self.model = best_model

        if args.save_model:
            pretrained_model_dir = os.path.join(args.method_output_dir, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)
    
    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'labeled':
            dataloader = self.train_labeled_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        eval_loss = 0
        nb_eval_steps = 0
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                
                features, logits = self.model(input_ids, segment_ids, input_mask, feature_ext = True)
                if mode == 'eval':
                    eval_loss = self.criterion(logits, label_ids)
                    
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, features))
                total_logits = torch.cat((total_logits, logits))
                
                nb_eval_steps += 1
                
        eval_loss = eval_loss / nb_eval_steps
        total_probs = F.softmax(total_logits.detach(), dim = 1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        
        outputs = {
            'feats': total_features.cpu().numpy(),
            'y_true': total_labels.cpu().numpy(),
            'y_pred': total_preds.cpu().numpy(),
            'loss': eval_loss
        }
        
        return outputs

    def predict_k(self, args, data):
     
        outputs = self.get_outputs(args, mode = 'train')
        feats, y_true = outputs['feats'], outputs['y_true']
        
        labeled_pos = list(np.where(y_true != -1)[0])
        labeled_feats = feats[labeled_pos]
        labeled_labels = y_true[labeled_pos]        
        unique_labeled_labels = np.unique(labeled_labels)
        
        labeled_centers = []
        for idx, label in enumerate(unique_labeled_labels):
            label_feats = labeled_feats[labeled_labels == label]
            labeled_centers.append(np.mean(label_feats, axis = 0))
        labeled_centers = np.array(labeled_centers)
        
        start = time.time()
        km = KMeans(n_clusters = data.num_labels, random_state = args.seed).fit(feats)
        km_centroids, y_pred = km.cluster_centers_, km.labels_
        end = time.time()
        self.logger.info('K-means used %s s', round(end - start, 2))
        
        DistanceMatrix = np.linalg.norm(labeled_centers[:,np.newaxis,:]-km_centroids[np.newaxis,:,:],axis=2) 
        row_ind, col_ind = linear_sum_assignment(DistanceMatrix)        
        alignment_labels = list(col_ind)

        cluster_mean_size = len(y_true) / data.num_labels
        print('cluster_mean_size:{}'.format(cluster_mean_size))
        
        pred_label_list = np.unique(y_pred)
       
        cnt = 0
        known_nums = []
        new_nums = []
        
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if label in alignment_labels:
                known_nums.append(num)
                continue
            
            new_nums.append(num)
            if num >= cluster_mean_size:
                cnt += 1

        print('known_nums:{}'.format(known_nums))
        print('new_nums:{}'.format(new_nums))
        
        num_labels = self.n_known_cls + cnt
        print('============Number of clusters is {}'.format(num_labels))

        return num_labels