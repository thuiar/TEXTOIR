import logging
import torch
import numpy as np
import os
import copy
import logging
import torch.nn.functional as F
import pandas as pd
import random
import math

from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model
from scipy.optimize import linear_sum_assignment
from collections import Counter
from utils.faster_mix_k_means_pytorch import K_Means
from utils.metrics import  clustering_accuracy_score


class PretrainDTCManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)

        loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_labeled_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader']
        
        args.num_labels = data.n_known_cls
        self.set_model_optimizer(args, data, model)
       
        self.loss_fct = loss_map[args.loss_fct]

        if args.pretrain:
            self.logger.info('Pre-raining start...')
            self.train(args, data)
            self.logger.info('Pre-training finished...')

        else:
            self.model = restore_model(self.model, os.path.join(args.method_output_dir, 'pretrain'))
            
        if args.cluster_num_factor > 1.0:
            self.num_labels = data.num_labels
            self.num_labels = self.predict_k(args, data, model)

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, data, 'bert', args.freeze_bert_parameters)   
        self.optimizer, self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_labeled_examples), args.train_batch_size, \
            args.num_pretrain_epochs, args.lr_pre, args.warmup_proportion)
        
        self.device = model.device

    def predict_k(self, args, data, model):

        loader = data.dataloader
        self.dtc_train_labeled_dataloader, self.dtc_train_unlabeled_dataloader, self.dtc_eval_dataloader, self.dtc_val_labeled_dataloader = \
            loader.train_labeled_outputs_dtc['loader'], loader.train_unlabeled_outputs_dtc['loader'], loader.eval_outputs_dtc['loader'], loader.val_labeled_outputs_dtc['loader']
        
        self.predict_model = model.set_model(args, data, 'bert')
        self.predict_optimizer, self.predict_scheduler = model.set_optimizer(self.predict_model, len(data.dataloader.train_labeled_examples_dtc), args.train_batch_size, \
            args.num_pretrain_epochs, args.lr_pre, args.warmup_proportion)

        self.logger.info("***** Running predict k *****")
        self.predict_model.to(self.device)

        wait = 0
        best_model = None
        best_eval_score = 0
        patient = 1
        acc_best = 0
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):

            self.predict_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.dtc_train_labeled_dataloader, desc="Iteration (labeled)")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    
                    loss = self.predict_model(input_ids, segment_ids, input_mask, label_ids, loss_fct = self.loss_fct, mode = "train")
                    
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.predict_optimizer.step()
                    self.predict_scheduler.step()
                    self.predict_optimizer.zero_grad()
    
            
            loss = tr_loss / nb_tr_steps
           
            self.logger.info("loss: %s", str(loss))
            self.predict_model.eval()
            total_logits = torch.empty((0, data.n_known_cls)).to(self.device)
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            for batch in tqdm(self.dtc_eval_dataloader, desc="Extracting representation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
                    logits, _ = self.predict_model(input_ids, segment_ids, input_mask)
                    total_logits = torch.cat((total_logits, logits))
                    total_labels = torch.cat((total_labels, label_ids))
                    
            probs, preds = F.softmax(total_logits, dim = 1).max(dim = 1)
            y_pred = preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            acc = clustering_accuracy_score(y_true, y_pred)

            self.logger.info("eval_results: %s", str(acc))
            if acc > acc_best:
                model_best = copy.deepcopy(self.predict_model)
                wait = 0
                acc_best = acc
            else:
                wait += 1
                if wait >= patient:
                    self.predict_model = model_best
                    break  

        max_cand_k = self.num_labels

        self.predict_model.eval()
        u_labels = torch.empty(0, dtype=torch.long).to(self.device)
        u_features = torch.empty((0, args.feat_dim)).to(self.device)
        with torch.set_grad_enabled(False):
            for batch in tqdm(self.dtc_train_unlabeled_dataloader, desc="Extracting Features"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.predict_model(input_ids, segment_ids, input_mask, feature_ext = True)
                u_features = torch.cat((u_features, features))
                u_labels = torch.cat((u_labels, label_ids))

        u_feats = u_features.cpu().numpy()
        u_labels = u_labels.cpu().numpy()

        self.predict_model.eval()
        l_labels = torch.empty(0, dtype=torch.long).to(self.device)
        l_features = torch.empty((0, args.feat_dim)).to(self.device)
        with torch.set_grad_enabled(False):
            for batch in tqdm(self.dtc_val_labeled_dataloader, desc="Extracting Features"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.predict_model(input_ids, segment_ids, input_mask, feature_ext = True)
                l_features = torch.cat((l_features, features))
                l_labels = torch.cat((l_labels, label_ids))

        l_feats = l_features.cpu().numpy()
        l_targets = l_labels.cpu().numpy()

        l_classes = set(l_targets) 
        split_ratio = 0.75
        num_lt_cls = int(round(len(l_classes) * split_ratio))

        lt_classes = set(random.sample(l_classes, num_lt_cls)) 
        lv_classes = l_classes - lt_classes

        lt_feats = np.empty((0, l_feats.shape[1]))
        lt_targets = np.empty(0)
        for c in lt_classes:
            lt_feats = np.vstack((lt_feats, l_feats[l_targets==c]))
            lt_targets = np.append(lt_targets, l_targets[l_targets==c])

        lv_feats = np.empty((0, l_feats.shape[1]))
        lv_targets = np.empty(0)
        for c in lv_classes:
            lv_feats = np.vstack((lv_feats, l_feats[l_targets==c]))
            lv_targets = np.append(lv_targets, l_targets[l_targets==c])

        cand_k = np.arange(max_cand_k)
        cvi_list = np.zeros(len(cand_k))
        acc_list = np.zeros(len(cand_k))
        u_num = len(u_labels)
        l_num = len(l_targets)
        cat_pred_list = np.zeros([len(cand_k),u_num+l_num])

        self.logger.info("estimating K ...")
        from sklearn.metrics import silhouette_score

        num_k = 10
        cnt = 0
        last_k = -1
        num_val_cls = data.dataloader.num_val_cls
        for i in range(len(cand_k)):
            cvi_list[i],  cat_pred_i = self.labeled_val_fun(np.concatenate((lv_feats, u_feats)), lt_feats, lt_targets, cand_k[i]+num_val_cls)
            cat_pred_list[i, :] = cat_pred_i
            print(cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
            lv_targets = np.array([int(num) for num in lv_targets])
            print(lv_targets)
            acc_list[i] = clustering_accuracy_score(lv_targets, cat_pred_i[len(lt_targets): len(lt_targets)+len(lv_targets)])
            best_k = self.get_best_k(cvi_list[:i+1], acc_list[:i+1], cat_pred_list[:i+1], l_num) + data.n_known_cls
            if best_k == last_k:
                cnt += 1
                if cnt >= num_k:
                    break
            else: 
                last_k = best_k
                cnt=0
                
            self.logger.info("current best K: %s", str(best_k))

        self.logger.info("best K: %s", str(best_k))

        return best_k

    def get_best_k(self, cvi_list, acc_list, cat_pred_list, l_num):
        min_max_ratio = 0.1
        idx_cvi = np.max(np.argwhere(cvi_list==np.max(cvi_list)))
        idx_acc = np.max(np.argwhere(acc_list==np.max(acc_list)))
        idx_best = int(math.ceil((idx_cvi+idx_acc)*1.0/2))
        cat_pred = cat_pred_list[idx_best, :]
        cnt_cat = Counter(cat_pred.tolist())
        cnt_l = Counter(cat_pred[:l_num].tolist())
        cnt_ul = Counter(cat_pred[l_num:].tolist())
        bin_cat = [x[1] for x in sorted(cnt_cat.items())]
        bin_l = [x[1] for x in sorted(cnt_l.items())]
        bin_ul = [x[1] for x in sorted(cnt_ul.items())]
        best_k = np.sum(np.array(bin_ul)/np.max(bin_ul).astype(float)>min_max_ratio)
        
        return best_k

    def labeled_val_fun(self, u_feats, l_feats, l_targets, k):
        if self.device=='cuda':
            torch.cuda.empty_cache()
        l_num=len(l_targets)
        kmeans = K_Means(k, pairwise_batch_size=256)
        kmeans.fit_mix(torch.from_numpy(u_feats).to(self.device), torch.from_numpy(l_feats).to(self.device), torch.from_numpy(l_targets).to(self.device))
        cat_pred = kmeans.labels_.cpu().numpy() 
        u_pred = cat_pred[l_num:]
        
        silh_score = silhouette_score(u_feats, u_pred)
        return silh_score, cat_pred 

    def train(self, args, data):    

        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):

            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration (labeled)")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, loss_fct = self.loss_fct, mode = "train")
                    
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                    
                    self.optimizer.zero_grad()
    
                    
            loss = tr_loss / nb_tr_steps

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = accuracy_score(eval_true, eval_pred)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
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

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_logits = torch.empty((0,args.num_labels)).to(self.device)
        total_features = torch.empty((0,args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                logits, probs = self.model(input_ids, segment_ids, input_mask)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred