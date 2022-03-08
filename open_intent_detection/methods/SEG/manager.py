import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
import pandas as pd
from torch import nn
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import trange, tqdm

from utils.functions import save_model
from utils.metrics import F_measure
from utils.functions import restore_model
from losses import loss_map

from sklearn.neighbors import LocalOutlierFactor


class SEGManager:

    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.model = model.model
        self.optimizer = model.optimizer
        self.scheduler = model.scheduler
        self.device = model.device

        self.data = data 
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader 
        self.test_dataloader = data.dataloader.test_loader

        if args.train:
            self.best_features = None
        else:
            restore_model(self.model, args.model_output_dir)
            self.best_features = np.load(os.path.join(args.method_output_dir, 'features.npy'))

    def get_class_feats(self, args, data):
        
        from dataloaders.bert_loader import convert_examples_to_features, InputExample
        from transformers import BertTokenizer
        
        known_labels = data.known_label_list
        examples = []
        for i, label in enumerate(known_labels):
            if args.dataset == 'stackoverflow':
                label = label.replace('-', ' ')
            else:
                label = label.replace('_', ' ')

            guid = "label-%s" % i
            examples.append(InputExample(guid=guid, text_a=label, text_b=None, label=None))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        max_label_length = max([len(label.replace('_', ' ').split()) for label in known_labels]) + 2

        if args.dataset == 'stackoverflow':
            max_label_length = max([len(tokenizer.tokenize(label.replace('-', ' '))) for label in known_labels]) + 2
        
        features = convert_examples_to_features(examples, None, max_label_length, tokenizer)
        
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

        class_feats = tuple((input_ids, segment_ids, input_mask))

        return class_feats

    def train(self, args, data):
        
        train_labels = [example.label for example in data.dataloader.train_labeled_examples]
        self.p_y = torch.tensor(np.unique(train_labels, return_counts=True)[1] / data.dataloader.num_train_examples)
        self.logger.info("Priori probability of each class = %s", self.p_y.numpy())
        if args.class_emb:
            class_feats = self.get_class_feats(args, data)
            self.class_feats = tuple(t.to(self.device) for t in class_feats)
            class_ids, class_segment, class_mask = self.class_feats

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

                with torch.set_grad_enabled(False):
                    class_emb = self.model(class_ids, class_segment, class_mask, feature_ext=True) if args.class_emb else None

                with torch.set_grad_enabled(True):

                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', device=self.device, class_emb=class_emb, p_y = self.p_y)
                    
                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            y_true, y_pred = self.get_outputs(args, data, self.eval_dataloader)

            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            
            eval_results = {
                'train_loss': loss,
                'eval_acc': eval_score,
                'best_acc':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                best_model = copy.deepcopy(self.model)
                wait = 0
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        
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
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask, p_y = self.p_y, device = self.device)

                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats 
        else:
            total_preds = torch.argmax(total_logits.detach(), dim=1)
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            
            if train_feats is not None:
                feats = total_features.cpu().numpy()
                y_pred = self.classify_lof(data, y_pred, train_feats, feats)
            
            return y_true, y_pred

    def test(self, args, data, show=False):
        train_feats = self.get_outputs(args, data, self.train_dataloader, get_feats = True)
        y_true, y_pred = self.get_outputs(args, data, self.test_dataloader, train_feats = train_feats)
        
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
    