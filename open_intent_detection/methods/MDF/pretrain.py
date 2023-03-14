import torch
import torch.nn.functional as F
import os
import copy
import logging
import torch.nn as nn

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from torch.utils.data import RandomSampler, DataLoader
from utils.functions import save_model, mask_tokens, restore_model
from transformers import AutoTokenizer

class PretrainManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):
        
        self.logger = logging.getLogger(logger_name)
        
        self.set_model_optimizer(args, data, model)
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]  
        self.best_eval_score = None
        
        if args.pretrain or (not os.path.exists(args.model_output_dir)):
            self.logger.info('Pre-raining start...')

            self.train(args, data)
            self.logger.info('Pre-training finished...')
        else:
            pretrained_model_dir = os.path.join(args.method_output_dir, 'pretrain')
            self.model = restore_model(self.model, pretrained_model_dir)

    def set_model_optimizer(self, args, data, model):
        args.backbone = 'bert_mdf_pretrain'
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device


    def train(self, args, data):
        self.model.train()
        
        wait = 0
        best_model = None
        best_eval_score = 0
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_model)
        
                
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0


            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
            
                input_ids, input_mask, segment_ids, label_ids = batch
                X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                mask_ids, mask_lb = mask_tokens(input_ids.cpu(), tokenizer)
                mask_input_ids = mask_ids.to(self.device)
                
                X_mlm = {"input_ids": mask_input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                
                with torch.set_grad_enabled(True):
                    labeled_logits = self.model(X)["logits"]
                    
                    loss_src = self.model.loss_ce(labeled_logits, label_ids)    
                    loss_mlm = self.model.mlmForward(X_mlm, mask_lb.to(self.device))   

                    lossTOT = loss_src + loss_mlm 
                    lossTOT.backward()
                    
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += lossTOT.item()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
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
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()
        
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                outputs = self.model(X)
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

  