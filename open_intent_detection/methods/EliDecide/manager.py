import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from .boundary import BoundaryLoss
from losses import loss_map
from utils.functions import save_model, euclidean_metric
from utils.metrics import F_measure
from utils.functions import restore_model, centroids_cal
from .pretrain import PretrainManager
from copy import deepcopy

class Manager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.data = data

        pretrain_model = PretrainManager(args, data, model)
        self.model = pretrain_model.model

        self.centroids = pretrain_model.centroids
        self.pretrain_best_eval_score = pretrain_model.best_eval_score

        self.device = self.model.device

        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.best_eval_score = None
        
        if args.train:
            self.delta = None
            self.delta_points = []

        else:
            self.model = restore_model(self.model, args.model_output_dir)
            self.delta = np.load(os.path.join(args.method_output_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(args.method_output_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)

    def train(self, args, data):
        self.model.eval()
        self.centroids, self.delta = centroids_cal(self.model, args, data, self.train_dataloader, self.device, need_delta=True)
        print(self.delta)


        self.boundary_model = BoundaryLoss(args, self.device)
        optimizer = torch.optim.Adam(self.boundary_model.parameters(), lr = args.lr_boundary, weight_decay = 0.01)
        
        
        best_eval_score, best_centroids = 0, None
        wait = 0
        best_boundary_model = None
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.eval()
            self.boundary_model.train()
            ploss, nloss, tr_loss = 0, 0, 0
            pnum, nnum = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True).detach()
                with torch.set_grad_enabled(True):
                    pos_loss, neg_loss, pos_num, neg_num, loss = self.boundary_model(features, self.centroids, self.delta, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ploss += pos_loss.item()
                    nloss += neg_loss.item()
                    pnum += pos_num.item()
                    nnum += neg_num.item()
                    tr_loss += loss.item()
                    
                    nb_tr_examples += features.shape[0]
                    nb_tr_steps += 1

            ploss = ploss / nb_tr_steps
            nloss = nloss / nb_tr_steps
            pnum = pnum / nb_tr_steps
            nnum = nnum / nb_tr_steps
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'pos_loss': ploss,
                'neg_loss': nloss,
                'pos_num': pnum,
                'neg_num': nnum,
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                wait = 0
                best_boundary_model = self.boundary_model
                best_eval_score = eval_score


            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break


        if best_eval_score > 0:
            self.boundary_model = best_boundary_model
            self.best_eval_score = best_eval_score

    def get_outputs(self, args, data, mode = 'eval', get_feats = False, pre_train= False, delta = None):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()
        self.boundary_model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                
                pooled_output = self.model(input_ids, segment_ids, input_mask, feature_ext=True)

                preds = self.open_classify(data, pooled_output)
                total_preds = torch.cat((total_preds, preds))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))


        feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if get_feats:
            return feats, y_true, y_pred
        else:
            return y_true, y_pred
        

    def open_classify(self, data, features):

        # _, logits = self.model(features)
        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        d = self.delta[preds]

        '''
        rotate_x = torch.bmm(self.Q[preds], features.unsqueeze(2)).squeeze(2)
        pos_mask = (rotate_x > 0).type(torch.cuda.FloatTensor)
        neg_mask = (rotate_x < 0).type(torch.cuda.FloatTensor)
        rotate_x = (self.Dpos[preds] * pos_mask + self.Dneg[preds] * neg_mask) * rotate_x
        '''

        # rotate_x = torch.bmm(self.rotate_matrix[preds], features.unsqueeze(2)).squeeze(2)
        rotate_x = self.boundary_model(features, self.centroids, self.delta, preds, get_rotate_x=True)
        
        euc_dis = torch.norm(rotate_x, 2, 1)
        preds[euc_dis >= d] = data.unseen_label_id

        return preds

    def test(self, args, data, show=True):
        
        if 'detailed_results' in args and args.detailed_results:
            feats, y_true, y_pred = self.get_outputs(args, data, mode = 'test', get_feats=True)
            np.save('feats.npy', feats)
            np.save('y_true.npy', y_true)
            np.save('y_pred.npy', y_pred)
        else:
            y_true, y_pred = self.get_outputs(args, data, mode = 'test')
        
        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        
        if show:
            self.logger.info("***** Test: Confusion Matrix *****")
            np.set_printoptions(linewidth=cm.shape[0] * 100, threshold=(cm.shape[0] + 10)**2)
            self.logger.info("%s", str(cm))
            self.logger.info("***** Test results *****")
            
            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

                
            self.logger.info("***** Rotate Matrix *****")
            self.logger.info("%s", str(self.boundary_model.get_rotate_matrix()))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        self.model.load_state_dict(pretrained_dict, strict=False)
