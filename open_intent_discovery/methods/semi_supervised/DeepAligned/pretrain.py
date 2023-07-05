import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging

from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm

from losses import loss_map
from utils.functions import save_model, restore_model
from sklearn.cluster import KMeans

class PretrainDeepAlignedManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)

        args.num_labels = data.n_known_cls

        self.set_model_optimizer(args, data, model)
        
        loader = data.dataloader
        self.train_labeled_dataloader = loader.train_labeled_outputs['loader']
        self.train_dataloader = loader.train_outputs['loader']
        self.eval_dataloader = loader.eval_outputs['loader']
        self.test_dataloader = loader.test_outputs['loader']

        self.loss_fct = loss_map[args.loss_fct]

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
        
        self.model = model.set_model(args, data, 'bert')  
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_labeled_examples), args.train_batch_size, \
            args.num_pretrain_epochs, args.lr_pre, args.warmup_proportion)
        
        self.device = model.device
        
    def train(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_labeled_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):

                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

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
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def predict_k(self, args, data):

        self.logger.info('Predict number of clusters start...')
        self.num_labels = data.num_labels
        feats = self.get_outputs(args, mode = 'train', get_feats = True)

        km = KMeans(n_clusters = self.num_labels, random_state =args.seed).fit(feats)
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