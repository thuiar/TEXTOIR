import torch
import copy
import pandas as pd
import logging

from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import LocalOutlierFactor
from utils.functions import restore_model, save_model
from utils.metrics import F_measure
from .KCL_utils import create_negative_dataset, generate_positive_sample, _prepare_inputs

class KCLManager:
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.set_model_optimizer(args, data, model)

        self.data = data 
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader= data.dataloader.eval_loader 
        self.test_dataloader = data.dataloader.test_loader

        self.negative_data = create_negative_dataset(self.train_dataloader)

        if not args.train:
            restore_model(self.model, args.model_output_dir)

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device


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

                positive_sample = None
                positive_sample = generate_positive_sample(self.negative_data,args, label_ids)
                positive_sample = _prepare_inputs(self.device, positive_sample)

                batch_dict = {"labels":label_ids,"input_ids":input_ids,"token_type_ids":segment_ids,"attention_mask":input_mask}
                outputs = self.model(batch_dict, mode='train', positive_sample=positive_sample)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                self.scheduler.step()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
            loss = tr_loss / nb_tr_steps
            y_true, y_pred = self.get_outputs(args, data, mode='eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score': best_eval_score
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


    def classify_lof(self, args, data, preds, train_feats, pred_feats):
        
        lof = LocalOutlierFactor(n_neighbors=args.n_neighbors, contamination = args.contamination, novelty=True, n_jobs=-1)
        lof.fit(train_feats)
        y_pred_lof = pd.Series(lof.predict(pred_feats))
        preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_label_id

        return preds

    def get_outputs(self, args, data, mode, get_feats = False, train_feats = None):
        
        if mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()
        
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_probs = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            batch_dict = {"labels":label_ids,"input_ids":input_ids,"token_type_ids":segment_ids,"attention_mask":input_mask}
            with torch.set_grad_enabled(False):
                output = self.model(batch_dict, mode='test')
                total_labels = torch.cat((total_labels,label_ids))
                total_probs = torch.cat((total_probs, output[0]))
                total_features = torch.cat((total_features, output[1]))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats 
        else:
            total_probs, y_pred = total_probs.max(dim = 1)
            y_pred = y_pred.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            
            if train_feats is not None:
                feats = total_features.cpu().numpy()
                y_pred = self.classify_lof(args, data, y_pred, train_feats, feats)
            
            return y_true, y_pred


    def test(self, args, data, show=False):
        
        train_feats = self.get_outputs(args, data, mode = 'train', get_feats = True)
        y_true, y_pred = self.get_outputs(args, data, mode = 'test', train_feats = train_feats)

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
