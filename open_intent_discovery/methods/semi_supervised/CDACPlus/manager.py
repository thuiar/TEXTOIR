import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from tqdm import trange, tqdm

from utils.metrics import clustering_score
from utils.functions import restore_model, save_model

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class CDACPlusManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        self.model = model.model
        self.optimizer1 = model.optimizer
        self.optimizer2 = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
            args.num_refine_epochs, args.lr, args.warmup_proportion)

        self.device = model.device 
        
        self.train_labeled_dataloader = data.dataloader.train_labeled_loader
        self.train_unlabeled_dataloader = data.dataloader.train_unlabeled_loader
        self.train_dataloader = data.dataloader.train_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader 

        if not args.train:
            self.model = restore_model(self.model, args.model_output_dir)

    def initialize_centroids(self, args, data):
        
        self.logger.info("Initialize centroids...")

        feats = self.get_outputs(args, mode = 'train_unlabeled', get_feats = True)
        km = KMeans(n_clusters=data.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)

        self.logger.info("Initialization finished...")

        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def train(self, args, data): 

        self.logger.info('Pairwise-similarity Learning begin...')
        
        u = args.u
        l = args.l
        eta = 0

        eval_pred_last = np.zeros_like(data.dataloader.eval_examples)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_labeled_dataloader, desc="Iteration (labeled)")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, u_threshold = u, l_threshold = l, mode = 'train')
                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer1.step()
                self.optimizer1.zero_grad() 

            train_labeled_loss = tr_loss / nb_tr_steps

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration (all train)")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, u_threshold = u, l_threshold = l, mode = 'train', semi = True)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer1.step()
                self.optimizer1.zero_grad()
            
            train_loss = tr_loss / nb_tr_steps

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']

            delta_label = np.sum(eval_pred != eval_pred_last).astype(np.float32) / eval_pred.shape[0]
            eval_pred_last = np.copy(eval_pred)

            train_results = {
                'u_threshold': round(u, 4),
                'l_threshold': round(l, 4),
                'train_labeled_loss': train_labeled_loss,
                'train_loss': train_loss,
                'delta_label': delta_label,
                'eval_score': eval_score
            }
            
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(train_results.keys()):
                self.logger.info("  %s = %s", key, str(train_results[key]))
            
            eta += 1.1 * 0.009
            u = 0.95 - eta
            l = 0.455 + eta * 0.1
            if u < l:
                break
        
        self.logger.info('Pairwise-similarity Learning finished...')

        self.refine(args, data)

    def refine(self, args, data):
        
        self.logger.info('Cluster refining begin...')
        self.initialize_centroids(args, data)

        best_model = None
        wait = 0
        train_preds_last = None
        best_eval_score = 0

        for epoch in range(args.num_refine_epochs):
            
            #evaluation
            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']

            #early stop
            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score
                self.model = best_model

            else:
                wait += 1
                if wait > args.wait_patient:
                    break

            #converge
            train_pred_logits = self.get_outputs(args, mode = 'train', get_logits = True)
            p_target = target_distribution(train_pred_logits)
            train_preds = train_pred_logits.argmax(1)

            delta_label = np.sum(train_preds != train_preds_last).astype(np.float32) / train_preds.shape[0]
            train_preds_last = np.copy(train_preds)

            if epoch > 0 and delta_label < 0.001:
                self.logger.info('Break at epoch: %s and delta_label: %f.', str(epoch + 1), round(delta_label, 2))
                break
            
            # Fine-tuning with auxiliary distribution
            self.model.train()
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                feats, logits = self.model(input_ids, segment_ids, input_mask, mode='finetune')
                kl_loss = F.kl_div(logits.log(), torch.Tensor(p_target[step * args.train_batch_size: (step + 1) * args.train_batch_size]).cuda())
                kl_loss.backward()

                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer2.step()
                self.optimizer2.zero_grad() 
            
            train_loss = tr_loss / nb_tr_steps
            eval_results = {
                'kl_loss': round(train_loss, 4), 
                'delta_label': delta_label.round(4),
                'eval_score': round(eval_score, 2),
                'best_eval_score': round(best_eval_score, 2)
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

        self.logger.info('Cluster refining finished...')

        if args.save_model:
            save_model(self.model, args.model_output_dir)
    
    def get_outputs(self, args,  mode = 'eval', get_feats = False, get_logits = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train_unlabeled':
            dataloader = self.train_unlabeled_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.num_labels)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)
    
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        elif get_logits:
            logits = total_logits.cpu().numpy()
            return logits

        else:
            total_preds = total_logits.argmax(1)
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def test(self, args, data):

        y_true, y_pred = self.get_outputs(args, mode = 'test')
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

        return test_results