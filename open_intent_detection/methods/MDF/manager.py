import torch
import torch.nn.functional as F
import logging
import os
import torch.nn as nn
import numpy as np
import copy
import json

from sklearn import svm 
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model
from utils.metrics import F_measure
from torch.utils.data import DataLoader
from .pretrain import PretrainManager
from transformers import AutoTokenizer

class MDFManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)
        self.set_model_optimizer(args, data, model)

        pretrain_manager = PretrainManager(args, data, model) 
        
        self.pretrained_model = pretrain_manager.model
        self.load_pretrained_model(self.pretrained_model.bert)

        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader


        self.loss_fct = loss_map[args.loss_fct]  
        self.best_eval_score = None
    
    def set_model_optimizer(self, args, data, model):
        args.backbone = 'bert_mdf'
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device

    def get_hidden_features(self, input_ids=None,  attention_mask=None, token_type_ids=None, labels=None,
        position_ids=None, head_mask=None, use_cls=False):
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        
        all_hidden_feats = outputs[1]   # list (13) of bs x length x hidden
        
        all_feature_list = []
        for i in range(len(all_hidden_feats)):
            if use_cls:
                pooled_feats = self.model.bert.pooler(all_hidden_feats[i]).detach()  # bs x max_len x 768 -> bs x 768
                # pooled_feats = all_hidden_feats[i][:,0].detach().data.cpu()  # bs x max_len x 768 -> bs x 768
                # print (pooled_feats.shape)
            else:
                pooled_feats = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()  # bs x max_len x 768 -> bs x 768
            all_feature_list.append(pooled_feats.data)   # 13 list of bs x 768
        return all_feature_list 
    

    def sample_X_estimator(self, use_cls=False):
        device = self.device 
        model = self.model 
        
        import sklearn.covariance
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        
        model.eval()
        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])
        
        for batch in tqdm(self.train_dataloader, desc="Iteration"):
            
            inputs = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                batch_all_features = self.get_hidden_features(*inputs, use_cls=use_cls)
                for i in range(num_layers):
                    all_layer_features[i].append(batch_all_features[i].cpu())  # save gpu memory
        
        mean_list = []
        precision_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.to(device))
            precision_list.append(temp_precision.to(device))

        return mean_list, precision_list

    def get_unsup_Mah_score(self, mode, sample_mean, precision, use_cls=False):
        device = self.device 
        model = self.model 

        model.eval()
        num_layers = 13
        total_mah_scores = []
        for i in range(num_layers):
            total_mah_scores.append([])

        
        if mode == 'train_labeled':
            dataloader = self.train_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        else:
            print('get_unsup_Mah_score error: unexpected mode')

        for batch in tqdm(dataloader, desc="Iteration"):
            inputs = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                batch_all_features = self.get_hidden_features(*inputs, use_cls=use_cls)
            
            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean
                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                total_mah_scores[i].extend(gaussian_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)
        return np.concatenate(total_mah_scores, axis=1)

    def train(self, args, data):
        pass

    def test(self, args, data, show=True):
        mean_list, precision_list = self.sample_X_estimator(args.use_cls)
        
        train_mah_vanlia = self.get_unsup_Mah_score('train_labeled', mean_list, precision_list, args.use_cls)[:, 1:]
        train_mah_scores = train_mah_vanlia
        

        c_lr = svm.OneClassSVM(nu=args.nuu, kernel=args.k)
        c_lr.fit(train_mah_scores)

        y_true, y_pred_ind = self.get_outputs(args, mode = 'test', model = self.pretrained_model, get_feats = False)
        test_total_mah_vanlia = self.get_unsup_Mah_score('test', mean_list, precision_list, args.use_cls)[:, 1:]
        y_pred_ood = c_lr.predict(test_total_mah_vanlia)

        y_pred = [args.unseen_label_id if y == -1 else y_pred_ind[i] for i, y in enumerate(y_pred_ood)]

        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        
        if show:
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
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                outputs = model(X)
                pooled_output = outputs["hidden_states"]
                logits = outputs["logits"]
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim=1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred
  
        
    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        self.model.bert.load_state_dict(pretrained_dict, strict=False)
