import torch
import torch.nn.functional as F
import os
import copy
import logging
import torch.nn as nn

from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from torch.utils.data import RandomSampler, DataLoader
from utils.functions import save_model, mask_tokens, set_seed, restore_model
from transformers import AutoTokenizer

class PretrainMTP_CLNNManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
    
        self.num_labels = args.num_labels = data.n_known_cls
        self.set_model_optimizer(args, data, model)
        
        
        loader = data.dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.eval_outputs['loader'], loader.test_outputs['loader']
        labeled_data = loader.train_labeled_outputs['data']

        sampler = RandomSampler(labeled_data)
        self.train_labeled_dataloader = DataLoader(labeled_data, sampler=sampler, batch_size = args.pretrain_batch_size, num_workers = args.num_workers, pin_memory = True)  

        self.criterion = loss_map['CrossEntropyLoss']
        self.temperature=0.07
        
        if args.pretrain:
            
            self.logger.info('Pre-raining start...')
            self.train(args, data)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.method_output_dir, 'pretrain'))

    def set_model_optimizer(self, args, data, model):
        
        args.backbone = 'bert_MTP_Pretrain'
        self.model = model.set_model(args, data, 'bert', args.freeze_train_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, len(data.dataloader.train_labeled_examples), args.pretrain_batch_size, \
            args.num_train_epochs, args.lr_pre, args.warmup_proportion)
        
        self.device = model.device 
        
    def train(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_model)
        mlm_iter = iter(self.train_dataloader) # mlm on semi-dataloader
                
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
        
            for step, batch in enumerate(tqdm(self.train_labeled_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
            
                labeled_input_ids, labeled_input_mask, labeled_segment_ids, labeled_label_ids = batch
                X = {"input_ids":labeled_input_ids, "attention_mask": labeled_input_mask, "token_type_ids": labeled_segment_ids}
                try:
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, _ = batch
                    X_mlm = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                except StopIteration:
                    mlm_iter = iter(self.train_dataloader)
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, _ = batch
                    X_mlm = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}

                mask_ids, mask_lb = mask_tokens(X_mlm['input_ids'].cpu(), tokenizer)
                input_ids = mask_ids.to(self.device)
                
                with torch.set_grad_enabled(True):
                    labeled_logits = self.model(X)["logits"]
                    
          
                    loss_src = self.model.loss_ce(labeled_logits, labeled_label_ids)    
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

        self.model.eval()
        
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
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
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

  