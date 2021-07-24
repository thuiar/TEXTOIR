import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm

from .boundary import BoundaryLoss
from losses import loss_map
from utils.functions import save_model
from utils.metrics import F_measure
from utils.functions import restore_model

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class ADBManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.model = model.model
        self.optimizer = model.optimizer
        self.device = model.device
        
        self.data = data
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader 
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]  
        
        if args.train:
            
            self.delta = None
            self.delta_points = []
            self.centroids = None

        else:

            self.model = restore_model(self.model, args.model_output_dir)
            self.delta = np.load(os.path.join(args.method_output_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(args.method_output_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)

    def pre_train(self, args, data):
        
        self.logger.info('Pre-training Start...')
        wait = 0
        best_model = None
        best_eval_score = 0

        if args.loss_fct == 'center_loss':
            from losses.CenterLoss import CenterLoss
            center_loss = CenterLoss(num_classes=data.num_labels, feat_dim=args.feat_dim, device = self.device)
            optimizer_centloss = torch.optim.Adam(center_loss.parameters(), lr = 0.05)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            
            if args.backbone == 'bert_disaware':
                self.centroids = self.centroids_cal(args, data)  

            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):

                    if args.backbone == 'bert_disaware':
                        
                        loss_fct = nn.CrossEntropyLoss()
                        loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct =  loss_fct, centroids = self.centroids)

                        if args.loss_fct == 'center_loss':
                            alpha = 0.05
                            features = self.model(input_ids, segment_ids, input_mask, label_ids, feature_ext = True)
                            loss = center_loss(features, label_ids) * alpha  + loss
                    else:

                        loss_fct = nn.CrossEntropyLoss()
                        loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = loss_fct)

                        if args.loss_fct == 'center_loss':
                            alpha = 0.05
                            features = self.model(input_ids, segment_ids, input_mask, label_ids, feature_ext = True)
                            loss = center_loss(features, label_ids) * alpha  + loss

                    self.optimizer.zero_grad()

                    if args.loss_fct == 'center_loss':
                        optimizer_centloss.zero_grad()

                    loss.backward()
                    self.optimizer.step()

                    if args.loss_fct == 'center_loss':
                        for param in center_loss.parameters():
                            param.grad.data *= (1./alpha)

                        optimizer_centloss.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, self.eval_dataloader, pre_train=True)
            eval_score = accuracy_score(y_true, y_pred)

            eval_results = {
                'train_loss': loss,
                'eval_acc': eval_score,
                'best_acc':best_eval_score,
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

        self.logger.info('Pre-training finished...')


    def train(self, args, data):  

        self.pre_train(args, data)   
        
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim).to(self.device)
        
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        best_eval_score, best_delta, best_centroids = 0, None, None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, self.eval_dataloader)
            eval_score = f1_score(y_true, y_pred, average='macro')

            eval_results = {
                'train_loss': loss,
                'eval_acc': eval_score,
                'best_acc':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:

                wait = 0
                best_delta = self.delta 
                best_centroids = self.centroids
                best_eval_score = eval_score

            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        self.delta = best_delta
        self.centroids = best_centroids

        if args.save_model:

            np.save(os.path.join(args.method_output_dir, 'centroids.npy'), self.centroids.detach().cpu().numpy())
            np.save(os.path.join(args.method_output_dir, 'deltas.npy'), self.delta.detach().cpu().numpy())
            

    def get_outputs(self, args, data, dataloader, get_feats = False, \
                                    pre_train= False, delta = None):
    
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                if args.backbone == 'bert_disaware':
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, centroids = self.centroids)
                else:
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask)
                
                if not pre_train:
                    preds = self.open_classify(data, pooled_output)
                    total_preds = torch.cat((total_preds, preds))

                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
    
            if pre_train:
                total_probs = F.softmax(total_logits.detach(), dim=1)
                total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred


    def open_classify(self, data, features):

        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_label_id

        return preds
    
    def test(self, args, data, show=True):
        
        y_true, y_pred = self.get_outputs(args, data, self.test_dataloader)
        
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

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        
        self.model.eval()
        centroids = torch.zeros(data.num_labels, args.feat_dim).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):

            for batch in self.train_dataloader:

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).to(self.device)
        
        return centroids

     




  

    
    
