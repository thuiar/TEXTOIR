import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
import pandas as pd

from sklearn.cluster import KMeans
from utils.metrics import clustering_score
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import trange, tqdm
from torch.utils.data import (DataLoader, SequentialSampler, RandomSampler, TensorDataset)
from losses import loss_map
from utils.functions import save_model, restore_model, set_seed
from utils.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from scipy.optimize import minimize_scalar
from functools import partial
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear_assignment

class GCDManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        set_seed(args.seed)
        
        loader = data.dataloader
        self.loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader']
        self.train_input_ids, self.train_input_mask, self.train_segment_ids, self.train_label_ids= \
            loader.train_outputs['input_ids'], loader.train_outputs['input_mask'], loader.train_outputs['segment_ids'], loader.train_outputs['label_ids']
        self.aug_train_dataloader = self.get_augment_dataloader(args, self.train_label_ids, data_aug = True)
        
        self.set_model_optimizer(args, data, model)
        self.num_labels = data.num_labels
        self.temperature=0.07
        self.sup_con_weight = 0.5
        self.loss_fct = loss_map[args.loss_fct]

        if not args.train:
            self.model = restore_model(self.model, args.model_output_dir)

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, data, 'bert', args.freeze_train_bert_parameters)
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_examples), args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device

    def batch_chunk(self, x):
        x1, x2 = torch.chunk(input=x, chunks=2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        return x1, x2

    def semisupvised_kmeans(self, args):
        # Semi-Kmeans
        feats, all_labels = self.get_outputs(args, mode = 'train')
        l_index = [k for k,i in enumerate(all_labels) if i !=-1]
        u_index = [k for k,i in enumerate(all_labels) if i ==-1]
        print('Fitting Semi-Supervised K-Means...')
        kmeans = SemiSupKMeans(k=self.num_labels, tolerance=1e-4, max_iterations=200, init='k-means++',
                        n_init=100, random_state=args.seed, n_jobs=None, pairwise_batch_size=1024, mode=None)
        u_feats = feats[u_index]
        l_feats = feats[l_index]
        l_targets = all_labels[l_index]
        u_targets = all_labels[u_index]
        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(self.device) for
                                            x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        self.semisupvised_kmeans_cluster = kmeans.cluster_centers_
         
    def train(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0
        criterion = loss_map['SupConLoss']
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_acc = 0
            
            for step, batch in enumerate(tqdm(self.aug_train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    
                    input_ids_a,  input_ids_b = self.batch_chunk(input_ids)
                    input_mask_a,  input_mask_b = self.batch_chunk(input_mask)
                    segment_ids_a,  segment_ids_b = self.batch_chunk(segment_ids)
                    label_ids = torch.chunk(input=label_ids, chunks=2, dim=1)[0][:, 0]

                    x_a = self.model(input_ids_a, segment_ids_a, input_mask_a, mode = 'train')               
                    x_b = self.model(input_ids_b, segment_ids_b, input_mask_b, mode = 'train')
                    
                    aug_mlp_outputs_a  = self.model.mlp_head(x_a)
                    aug_mlp_outputs_b  = self.model.mlp_head(x_b)

                    norm_logits = F.normalize(aug_mlp_outputs_a)
                    norm_aug_logits = F.normalize(aug_mlp_outputs_b)
                   
                    contrastive_feats = torch.cat((norm_logits, norm_aug_logits))
                    contrastive_logits, contrastive_labels = self.info_nce_logits(features=contrastive_feats)   
                    contrastive_loss = self.loss_fct(contrastive_logits, contrastive_labels)

                    mask_lab = torch.from_numpy(np.array([0 if i ==-1 else 1 for i in label_ids])).bool()
                    f1, f2 = [f[mask_lab] for f in contrastive_feats.chunk(2)]
                    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    sup_con_labels = label_ids[mask_lab]
                    sup_loss = criterion(features = sup_con_feats, labels = sup_con_labels, device = self.device)  
                    
                    loss =  self.sup_con_weight * sup_loss  + (1 - self.sup_con_weight) * contrastive_loss 
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                   

            train_loss = tr_loss / nb_tr_steps
            features, y_true  = self.get_outputs(args, mode = 'eval')
            km = KMeans(n_clusters = int(data.n_known_cls), random_state=args.seed).fit(features)
            y_pred = km.labels_
            eval_score = clustering_score(y_true, y_pred)
        
            eval_results = {
                'train_loss': train_loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
            }
            
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score['ACC'] > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score['ACC']
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.logger.info('GCD training finished...')
        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_dir)
    
        self.semisupvised_kmeans(args)

    def get_outputs(self, args, mode):
        if mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
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
        
        return  feats, y_true

    def info_nce_logits(self, features):

        b_ = 0.5 * int(features.size(0))

        labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
       
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
      
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
    def get_augment_dataloader(self, args, pseudo_labels, data_aug = False):

        train_input_ids = self.train_input_ids.unsqueeze(1)
        train_input_mask = self.train_input_mask.unsqueeze(1)
        train_segment_ids = self.train_segment_ids.unsqueeze(1)
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)

        train_input_ids = torch.cat(([train_input_ids, train_input_ids]), dim = 1)
        train_input_mask = torch.cat(([train_input_mask, train_input_mask]), dim = 1)
        train_segment_ids = torch.cat(([train_segment_ids, train_segment_ids]), dim = 1)
        train_label_ids = torch.cat(([train_label_ids, train_label_ids]), dim = 1)

        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
        
        label_len = len(self.loader.train_labeled_examples)
        unlabelled_len = len(self.loader.train_unlabeled_examples)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(self.loader.train_examples))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(self.loader.train_examples))

        
        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

        return train_dataloader

    def test(self, args, data):
        
        feats, y_true = self.get_outputs(args, mode = 'test')

        centers = self.semisupvised_kmeans_cluster 
        print("self.semisupvised_kmeans_cluster", self.semisupvised_kmeans_cluster)
        dis = (torch.from_numpy(feats).to(self.device).unsqueeze(dim=1)-centers.unsqueeze(dim=0))**2
        dis = dis.sum(dim = -1)
        u_mindist, y_pred = torch.min(dis, dim=1)
        y_pred = y_pred.cpu().numpy()

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
