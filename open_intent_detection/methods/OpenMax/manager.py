from importlib import import_module
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch import nn
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import trange, tqdm

from .openmax_utils import recalibrate_scores, weibull_tailfitting, compute_distance
from utils.metrics import F_measure
from losses import loss_map

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

        
class OpenMaxManager:
    
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
            self.weibull_model = None
            
        else:
            model_file = os.path.join(args.model_output_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(model_file))
            self.model.to(self.device)


    def train(self, args, data):     
        
        best_model = None
        wait = 0
        best_eval_score = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', loss_fct=self.loss_fct)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)

            y_true, y_pred = self.get_outputs(args, data, self.eval_dataloader) 
            eval_score = accuracy_score(y_true, y_pred)
            print('eval_score',eval_score)
            
            
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
            self.model.save_pretrained(args.model_output_dir, save_config=True)


    def get_outputs(self, args, data, dataloader, get_feats = False, compute_centroids=False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        
        centroids = torch.zeros(data.num_labels, data.num_labels).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)

                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, pooled_output))

                if compute_centroids:
                    for i in range(len(label_ids)):
                        centroids[label_ids[i]] += logits[i]  

        if get_feats:

            feats = total_features.cpu().numpy()
            return feats 

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            y_prob = total_probs.cpu().numpy()

            if compute_centroids:
                
                centroids /= torch.tensor(self.class_count(y_true)).float().unsqueeze(1).to(self.device)
                centroids = centroids.detach().cpu().numpy()

                y_logit = total_logits.cpu().numpy()
                
                mean_vecs, dis_sorted = self.cal_vec_dis(args, data, centroids, y_logit, y_true)
                weibull_model = weibull_tailfitting(mean_vecs, dis_sorted, data.num_labels, tailsize = args.weibull_tail_size)
                
                return weibull_model

            else:

                if self.weibull_model is not None:

                    y_logit = total_logits.cpu().numpy()
                    y_pred = self.classify_openmax(args, data, len(y_true), y_prob, y_logit)

                
                return y_true, y_pred

    def test(self, args, data, show = False):
            
        self.weibull_model = self.get_outputs(args, data, self.train_dataloader, compute_centroids=True)

        y_true, y_pred = self.get_outputs(args, data, self.test_dataloader)
        cm = confusion_matrix(y_true,y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc

        if show:
            print('cm',cm)
            print('test_results', test_results)

        return test_results
    
    def classify_openmax(self, args, data, num_samples, y_prob, y_logit):
            
        y_preds = []
        cnt = 0

        for i in range(num_samples):

            textarr = {}
            textarr['scores'] = y_prob[i]
            textarr['fc8'] = y_logit[i]
            openmax, softmax = recalibrate_scores(self.weibull_model, data.num_labels, textarr, alpharank=min(args.alpharank, data.num_labels))
            openmax = np.array(openmax)
            pred = np.argmax(openmax)
            max_prob = max(openmax)
            
            if pred == data.unseen_label_id:
                cnt += 1

            if max_prob < args.threshold:
                pred = data.unseen_label_id

            y_preds.append(pred)    

        return y_preds

    def cal_vec_dis(self, args, data, centroids, y_logit, y_true):

        mean_vectors = [x for x in centroids]

        dis_all = []
        for i in range(data.num_labels):
            arr = y_logit[y_true == i]
            dis_all.append(self.get_distances(args, arr, mean_vectors[i]))

        dis_sorted = [sorted(x) for x in dis_all]

        return mean_vectors, dis_sorted    
    
    def get_distances(self, args, arr, mav):

        pre = []
        for i in arr:
            pre.append(compute_distance(i, mav, args.distance_type))

        return pre

    def class_count(self, labels):

        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)

        return class_data_num
        


    



  

    
    
