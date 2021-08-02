import logging
import torch
import os
import copy
import torch.nn.functional as F

from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix

from losses import loss_map
from .pretrain import PretrainKCLManager
from utils.functions import restore_model, save_model
from utils.metrics import clustering_score

class KCLManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        backbone = args.backbone
        self.optimizer = model.optimizer
        self.device = model.device
        
        self.train_dataloader = data.dataloader.train_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        pretrain_manager = PretrainKCLManager(args, data, model)  
        self.loss_fct = loss_map[args.loss_fct]

        if args.train:
            
            self.logger.info('Pre-raining start...')
            pretrain_manager.train(args, data)
            self.logger.info('Pre-training finished...')

            self.pretrained_model = pretrain_manager.model

            args.num_labels = data.num_labels
            args.backbone = backbone
            self.model = model.set_model(args, data, 'bert')

        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.model = restore_model(self.model, args.model_output_dir)

    def train(self, args, data): 

        best_model = None
        wait = 0
        best_eval_score = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(self.train_dataloader, desc="Iteration"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                simi = self.prepare_task_target(batch, self.pretrained_model)
                
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = 'train', simi = simi, loss_fct = self.loss_fct)
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, mode = 'eval')

            eval_score = clustering_score(y_true, y_pred)['NMI']
            eval_results = {
                'train_loss': tr_loss,
                'eval_score': eval_score,
                'best_score': best_eval_score,
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
            
            #########debug
            # y_true, y_pred, feats = self.get_preds_labels(data.test_dataloader, self.model)
            # results = clustering_score(y_true, y_pred)
            # print('test_score', results)

        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):

                features, logits = self.model(input_ids, segment_ids, input_mask)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, features))

        if get_feats:

            feats = total_features.cpu().numpy()
            return feats

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_true = total_labels.cpu().numpy()
            y_pred = total_preds.cpu().numpy()

            return y_true, y_pred
    
    def prepare_task_target(self, batch, model):

        model.eval()
        input_ids, input_mask, segment_ids, label_ids = batch
        _, logits = model(input_ids, segment_ids, input_mask)
        probs = F.softmax(logits,dim=1)

        target = torch.argmax(probs,dim=1)
        target = target.float()
        target[target == 0] = -1

        return target.detach()

    def test(self, args, data):
        
        y_true, y_pred = self.get_outputs(args, mode = 'test')
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results
