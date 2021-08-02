import logging
import numpy as np
import copy
import torch
import os
import torch.nn.functional as F
from losses import loss_map
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from utils.functions import save_model

def Class2Simi(x,mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

class PretrainKCLManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        args.backbone = 'bert_KCL_simi'
        args.num_labels = 2
        self.model = model.set_model(args, data, 'bert')
        self.optimizer = model.set_optimizer(self.model, len(data.dataloader.train_labeled_examples), args.train_batch_size, \
            args.num_pretrain_epochs, args.lr_pre, args.warmup_proportion)
                    
        self.device = model.device
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader
        
        self.loss_fct = loss_map[args.pretrain_loss_fct]

    def train(self, args, data):  

        wait = 0
        best_model = None
        best_eval_score = 0
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):

            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                train_target = Class2Simi(label_ids, mode='cls').detach()

                loss = self.model(input_ids, segment_ids, input_mask, train_target, loss_fct = self.loss_fct, mode = 'train')
                
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1 
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            loss = tr_loss / nb_tr_steps
            
            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = round(accuracy_score(eval_true, eval_pred) * 100, 2)

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
        
    def get_outputs(self, args, mode = 'eval'):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0,args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            target = Class2Simi(label_ids, mode='cls').detach()

            with torch.set_grad_enabled(False):

                features, logits = self.model(input_ids, segment_ids, input_mask, mode = 'eval')
                total_labels = torch.cat((total_labels, target))
                total_logits = torch.cat((total_logits, logits))
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        return y_true, y_pred
