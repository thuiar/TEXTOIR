import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from utils.metrics import F_measure
from utils.functions import restore_model
from .pretrain import PretrainManager
from losses.ARPLoss import ARPLoss

class ARPLManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        pretrain_model = PretrainManager(args, data, model)
        self.model = pretrain_model.model
        self.pretrain_best_eval_score = pretrain_model.best_eval_score

        self.device = pretrain_model.device
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        # self.loss_fct = loss_map[args.loss_fct]
        self.best_eval_score = None
        
        if not args.train:
            self.model = restore_model(self.model, args.model_output_dir)

    def train(self, args, data):  
        self.arpl_criterion = ARPLoss(args)
        self.arpl_criterion.to(self.device)

        best_eval_score = 0
        wait = 0
        params_list = [{'params': self.arpl_criterion.parameters()}]
        optimizer = torch.optim.Adam(params_list, lr=args.lr_2)
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    logits, loss = self.arpl_criterion(features, labels=label_ids)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += features.shape[0]
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                wait = 0
                best_eval_score = eval_score
            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        if best_eval_score > 0:
            self.best_eval_score = best_eval_score

    def get_outputs(self, args, data, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                pooled_output = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                logits, loss = self.arpl_criterion(pooled_output)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            total_maxprobs_, total_preds_ = total_logits.max(dim=1)
            y_prob = total_maxprobs.cpu().numpy()

            y_true = total_labels.cpu().numpy()
            y_pred = total_preds.cpu().numpy()

            if mode == 'test':
                in_logits = []
                out_logits = []
                for ind, logit in enumerate(total_logits.detach().cpu().numpy()):
                    if y_true[ind] == data.unseen_label_id:
                        in_logits.append(logit)
                    else:
                        out_logits.append(logit)
                
                y_pred[y_prob < args.threshold] = data.unseen_label_id
                np.save(os.path.join(args.method_output_dir, 'y_prob.npy'), y_prob)
                return y_true, y_pred, in_logits, out_logits

        return y_true, y_pred
    
    def test(self, args, data, show=True):
        y_true, y_pred, in_logits, out_logits = self.get_outputs(args, data, mode = 'test')

        x1, x2 = np.max(in_logits, axis=1), np.max(out_logits, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        test_results['lr_2'] = args.lr_2
        test_results['temp'] = args.temp
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")

        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results
