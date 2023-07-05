import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import faiss
import time 

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment
from losses import loss_map
from utils.functions import save_model, restore_model
from transformers import BertTokenizer

from utils.metrics import clustering_score
from utils.functions import set_seed, view_generator
from losses import loss_map
from .pretrain import PretrainUnsupUSNIDManager

class UnsupUSNIDManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        pretrain_manager = PretrainUnsupUSNIDManager(args, data, model)
    
        set_seed(args.seed)
        self.logger = logging.getLogger(logger_name)

        loader = data.dataloader
        self.train_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.test_outputs['loader']

        self.train_outputs = loader.train_outputs
        self.criterion = loss_map['CrossEntropyLoss']
        self.contrast_criterion = loss_map['SupConLoss']
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)
        
        if args.pretrain:
            self.pretrained_model = pretrain_manager.model
            
            self.set_model_optimizer(args, data, model, pretrain_manager)
            self.load_pretrained_model(args, self.pretrained_model)
            
        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))   
            self.set_model_optimizer(args, data, model, pretrain_manager)
            
            if args.train:
                self.load_pretrained_model(args, self.pretrained_model)
            else:
                self.model = restore_model(self.model, args.model_output_dir)   
    
    def set_model_optimizer(self, args, data, model, pretrain_manager):
        
        args.num_labels = self.num_labels = data.num_labels
        self.model = model.set_model(args, data, 'bert', args.freeze_train_bert_parameters)     
        self.optimizer , self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
            args.num_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device
        
    def faiss_kmeans(self, args, x, init_centroids = None):
        
        d = x.shape[1]
        
        km = faiss.Kmeans(x.shape[1], self.num_labels, niter = 300, verbose = False, nredo = 10, gpu = 1, seed = args.seed)
        km.train(x, init_centroids = init_centroids)
        
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(d)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        centroids = km.centroids
        gpu_index_flat.add(centroids)
        distances, labels = gpu_index_flat.search(x, 1)
        
        return centroids, labels.ravel()
    
    def clustering(self, args, init = 'k-means++'):
        
        outputs = self.get_outputs(args, mode = 'train', model = self.model)
        feats = outputs['feats']
        y_true = outputs['y_true']
        
        if init == 'k-means++':
            
            self.logger.info('Initializing centroids with K-means++...')
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = 'k-means++').fit(feats) 
            
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            end = time.time()
            self.logger.info('K-means++ used %s s', round(end - start, 2))   
            
        elif init == 'centers':
            
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = self.centroids).fit(feats)
            km_centroids, assign_labels = km.cluster_centers_, km.labels_ 
            end = time.time()
            self.logger.info('K-means used %s s', round(end - start, 2))

        self.centroids = km_centroids
        pseudo_labels = torch.tensor(assign_labels, dtype=torch.long)      
        
        return outputs, km_centroids, y_true, assign_labels, pseudo_labels
                      
    def train(self, args, data): 

        self.centroids = None
        last_preds = None
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            
            outputs, km_centroids, y_true, assign_labels, pseudo_labels = self.clustering(args, init = init_mechanism)

            current_preds = pseudo_labels.numpy()
            delta_label = np.sum(current_preds != last_preds).astype(np.float32) / current_preds.shape[0] 
            last_preds = np.copy(current_preds)
            
            if epoch > 0:
                
                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Training Loss: %f', np.round(tr_loss, 5))
                self.logger.info('Delta Label: %f', delta_label)
                
                if delta_label < args.tol:
                    self.logger.info('delta_label %s < %f', delta_label, args.tol)  
                    self.logger.info('Reached tolerance threshold. Stop training.')
                    break                   
            
            self.train_outputs['label_ids'] = pseudo_labels

            pseudo_train_dataloader = self.get_augment_dataloader(args, self.train_outputs, pseudo_labels)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(pseudo_train_dataloader, desc="Training(All)"):
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):

                    input_ids_a, input_ids_b = self.batch_chunk(input_ids)
                    input_mask_a, input_mask_b = self.batch_chunk(input_mask)
                    segment_ids_a, segment_ids_b = self.batch_chunk(segment_ids)
                    label_ids = torch.chunk(input=label_ids, chunks=2, dim=1)[0][:, 0]
                          
                    aug_mlp_output_a, aug_logits_a = self.model(input_ids_a, segment_ids_a, input_mask_a, mode = 'train')               
                    aug_mlp_output_b, aug_logits_b = self.model(input_ids_b, segment_ids_b, input_mask_b, mode = 'train')

                    if not args.wo_ce:
                        loss_ce = 0.5 * (self.criterion(aug_logits_a, label_ids) + self.criterion(aug_logits_b, label_ids))

                    norm_logits = F.normalize(aug_mlp_output_a)
                    norm_aug_logits = F.normalize(aug_mlp_output_b)
                    
                    contrastive_feats = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    loss_contrast = self.contrast_criterion(contrastive_feats, labels = label_ids, temperature = args.train_temperature, device = self.device)
                    
                    
                    if not args.wo_ce:
                        loss = loss_contrast + loss_ce
                    else:
                        loss = loss_contrast
                    
                    self.optimizer.zero_grad()
                    loss.backward()

                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                
            tr_loss = tr_loss / nb_tr_steps
                
        if args.save_model:
            save_model(self.model, args.model_output_dir)
              
    def test(self, args, data):
        
        outputs = self.get_outputs(args, mode = 'test', model = self.model)
        feats = outputs['feats']
        y_true = outputs['y_true']

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = self.centroids).fit(feats) 
       
        y_pred = km.labels_
        
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

    def get_outputs(self, args, mode, model):
        
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits = model(input_ids, segment_ids, input_mask, feature_ext=True)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        
        y_logits = total_logits.cpu().numpy()
        
        outputs = {
            'y_true': y_true,
            'y_pred': y_pred,
            'logits': y_logits,
            'feats': feats
        }
        return outputs

    def load_pretrained_model(self, args, pretrained_model):
        
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['mlp_head.bias','mlp_head.0.bias',  'classifier.weight', 'classifier.bias', 'mlp_head.0.weight', 'mlp_head.weight'] 
        
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def batch_chunk(self, x):
        x1, x2 = torch.chunk(input=x, chunks=2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        return x1, x2
    
    def get_augment_dataloader(self, args, train_outputs, pseudo_labels = None):
        
        input_ids = train_outputs['input_ids']
        input_mask = train_outputs['input_mask']
        segment_ids = train_outputs['segment_ids']
        if pseudo_labels is None:
            pseudo_labels = train_outputs['label_ids']
        
        input_ids_a, input_mask_a = self.generator.random_token_erase(input_ids, input_mask)
        input_ids_b, input_mask_b = self.generator.random_token_erase(input_ids, input_mask)
        
        train_input_ids = torch.cat(([input_ids_a.unsqueeze(1), input_ids_b.unsqueeze(1)]), dim = 1)
        train_input_mask = torch.cat(([input_mask_a.unsqueeze(1), input_mask_a.unsqueeze(1)]), dim = 1)
        train_segment_ids = torch.cat(([segment_ids.unsqueeze(1), segment_ids.unsqueeze(1)]), dim = 1)
        
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        train_label_ids = torch.cat(([train_label_ids, train_label_ids]), dim = 1)
        
        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)

        sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

        return train_dataloader