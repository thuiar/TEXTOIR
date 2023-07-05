import logging
import copy
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd

from .pretrain import PretrainDTCManager
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from utils.metrics import clustering_score, clustering_accuracy_score
from utils.functions import save_model, restore_model, set_seed
from utils.faster_mix_k_means_pytorch import K_Means
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

class DTCManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        pretrain_manager = PretrainDTCManager(args, data, model)  
        
        set_seed(args.seed)
        self.logger = logging.getLogger(logger_name)
        
        loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_unlabeled_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader'] 
 
        if args.pretrain:
                     
            self.pretrained_model = pretrain_manager.model
            self.set_model_optimizer(args, data, model, pretrain_manager)
            self.load_pretrained_model(self.pretrained_model)
        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.set_model_optimizer(args, data, model, pretrain_manager)
            
            if args.train:
                self.load_pretrained_model(self.pretrained_model)
            else:
                self.model = restore_model(self.model, args.model_output_dir)

    def set_model_optimizer(self, args, data, model, pretrain_manager):

        if args.cluster_num_factor > 1:
            args.num_labels = self.num_labels = pretrain_manager.num_labels
        else:
            args.num_labels = self.num_labels = data.num_labels
            
        num_train_examples = len(data.dataloader.train_unlabeled_examples)

        self.model = model.set_model(args, data, 'bert', args.freeze_bert_parameters)

        self.warmup_optimizer, self.warmup_scheduler = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_warmup_train_epochs, args.lr, args.warmup_proportion)

        self.optimizer, self.scheduler = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device
        self.model.to(self.device)

    def initialize_centroids(self, args):

        self.logger.info("Initialize centroids...")

        feats = self.get_outputs(args, mode = 'train', get_feats = True)
        km = KMeans(n_clusters=args.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)
        self.logger.info("Initialization finished...")
        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def warmup_train(self, args):
        
        probs = self.get_outputs(args, mode = 'train', get_probs = True)
        p_target = target_distribution(probs)

        for epoch in trange(int(args.num_warmup_train_epochs), desc="Warmup_Epoch"):

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Warmup_Training")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits, q = self.model(input_ids, segment_ids, input_mask)
                loss = F.kl_div(q.log(),torch.Tensor(p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.warmup_optimizer.step()
                self.warmup_scheduler.step()
                self.warmup_optimizer.zero_grad()       

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            eval_results = {
                'loss': tr_loss, 
                'eval_score': round(eval_score, 2)
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
        
        return p_target
    
    def get_outputs(self, args, mode = 'eval', get_feats = False, get_probs = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.num_labels)).to(self.device)
        total_probs = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                logits, probs  = self.model(input_ids, segment_ids, input_mask)
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, logits))
                total_probs = torch.cat((total_probs, probs))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        elif get_probs:
            return total_probs.cpu().numpy()

        else:
            total_preds = total_probs.argmax(1)
            y_pred = total_preds.cpu().numpy()

            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def train(self, args, data): 

        self.initialize_centroids(args)

        self.logger.info('WarmUp Training start...')
        self.p_target = self.warmup_train(args)
        self.logger.info('WarmUp Training finished...')
          
        ntrain = len(data.dataloader.train_unlabeled_examples)
        Z = torch.zeros(ntrain, args.num_labels).float().to(self.device)       
        z_ema = torch.zeros(ntrain, args.num_labels).float().to(self.device)        
        z_epoch = torch.zeros(ntrain, args.num_labels).float().to(self.device) 

        best_model = None
        best_eval_score = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits, q = self.model(input_ids, segment_ids, input_mask)
                z_epoch[step * args.train_batch_size: (step+1) * args.train_batch_size, :] = q
                kl_loss = F.kl_div(q.log(), torch.tensor(self.p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))
                
                kl_loss.backward() 
                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad() 
            
            z_epoch = torch.tensor(self.get_outputs(args, mode = 'train', get_probs = True)).to(self.device)
            Z = args.alpha * Z + (1. - args.alpha) * z_epoch
            z_ema = Z * (1. / (1. - args.alpha ** (epoch + 1)))

            if epoch % args.update_interval == 0:
                self.logger.info('updating target ...')
                self.p_target = target_distribution(z_ema).float().to(self.device) 
                self.logger.info('updating finished ...')

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            train_loss = tr_loss / nb_tr_steps

            eval_results = {
                'train_loss': train_loss, 
                'best_eval_score': best_eval_score,
                'eval_score': round(eval_score, 2),
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
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):
    
        y_true, y_pred = self.get_outputs(args,mode = 'test')
        test_results = clustering_score(y_true, y_pred) 
        cm = confusion_matrix(y_true,y_pred) 
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        if args.cluster_num_factor > 1:
            test_results['estimate_k'] = args.num_labels

        return test_results

    def load_pretrained_model(self, pretrained_model):
    
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['cluster_layer', 'classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
