from importlib import import_module
import torch
import torch.nn.functional as F
import copy
import logging
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import restore_model, save_model
from utils.metrics import F_measure


class MixUpManager:
    
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
        self.neg_dataloader = data.dataloader.neg_loader

        self.loss_fct = loss_map[args.loss_fct]
        
        args.oos_label_id = data.unseen_label_id

        if not args.train:
            restore_model(self.model, args.model_output_dir)


    def get_outputs(self, args, data, mode = 'eval', get_feats = False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels+1)).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits, labels = self.model(input_ids, segment_ids, input_mask, label_ids, mode=mode)
                total_labels = torch.cat((total_labels, labels))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_prob = total_maxprobs.cpu().numpy()

            y_true = total_labels.cpu().numpy()
            y_pred = total_preds.cpu().numpy()

        return y_true, y_pred

    def test(self, args, data, show=False):
        y_true, y_pred = self.get_outputs(args, data, mode = 'test')

        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")

        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results


    def train(self, args, data):     

        self.logger.info('Training Start...')
        best_model = None
        best_eval_score = 0

        wait = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, (pst, ngt) in enumerate(zip(self.train_dataloader, self.neg_dataloader)):
                input_ids = torch.cat([pst[0], ngt[0]], dim=0).to(self.device)
                input_mask = torch.cat([pst[1], ngt[1]], dim=0).to(self.device)
                segment_ids = torch.cat([pst[2], ngt[2]], dim=0).to(self.device)
                label_ids = torch.cat([pst[3], ngt[3]], dim=0).to(self.device)

                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', loss_fct=self.loss_fct)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_acc': eval_score,
                'best_acc': best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score >= best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score 
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model 

        if args.save_model:
            save_model(self.model, args.model_output_dir)
        self.logger.info('Training finished...')
