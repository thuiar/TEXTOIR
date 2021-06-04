from importlib import import_module
import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import pandas as pd
from torch import nn
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import trange, tqdm

from losses import loss_map
from utils.metrics import F_measure

from sklearn.neighbors import LocalOutlierFactor


class DeepUnkManager:
    
    def __init__(self, args, data, model):

        self.model = model.model 
        self.optimizer = model.optimizer
        self.device = model.device

        self.data = data 
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader 
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]

        if args.train:

            self.best_features = None

        else:
            
            model_file = os.path.join(args.model_output_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(model_file))
            self.model.to(self.device)
            
            self.best_features = np.load(os.path.join(args.method_output_dir, 'features.npy'))

    def train(self, args, data):     

        best_model = None
        best_eval_score = 0
        wait = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', loss_fct = self.loss_fct)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)

            train_feats = self.get_outputs(args, data, self.train_dataloader, get_feats = True)

            y_true, y_pred = self.get_outputs(args, data, self.eval_dataloader, train_feats = train_feats)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            print('eval_score', eval_score)
            
            if eval_score >= best_eval_score:
                
                self.best_feats = train_feats
                best_eval_score = eval_score
                best_model = copy.deepcopy(self.model)
                wait = 0

            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model 

        np.save(os.path.join(args.method_output_dir, 'features.npy'), self.best_feats)
        
        if args.save_model:
            save_model(self.model, args.model_output_dir)


    def classify_lof(self, data, preds, train_feats, pred_feats):
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination = 0.05, novelty=True, n_jobs=-1)
        lof.fit(train_feats)
        y_pred_lof = pd.Series(lof.predict(pred_feats))
        preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_label_id

        return preds

    def get_outputs(self, args, data, dataloader, get_feats = False, train_feats = None):
    
        self.model.eval()
        
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)

                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats 
        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            
            if train_feats is not None:
                feats = total_features.cpu().numpy()
                y_pred = self.classify_lof(data, y_pred, train_feats, feats)
            
            return y_true, y_pred

    def test(self, args, data, show=False):
    
        y_true, y_pred = self.get_outputs(args, data, self.test_dataloader, train_feats = self.best_feats)
        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        
        if show:
            print('cm',cm)
            print('results', test_results)

        return test_results
    

  

    
    
