import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import time

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model, view_generator
from sklearn.cluster import KMeans

class PretrainUnsupUSNIDManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        
        args.num_labels = data.num_labels
        self.set_model_optimizer(args, data, model)

        loader = data.dataloader
        self.train_outputs = loader.train_outputs
        self.contrast_criterion = loss_map['SupConLoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)

        if args.pretrain:
            
            self.logger.info('Pre-raining start...')
            self.train(args, data)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.method_output_dir, 'pretrain'))

        if args.cluster_num_factor > 1:
            self.num_labels = data.num_labels
            self.num_labels = self.predict_k(args, data) 
            
        self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()

    def predict_k(self, args, data):
        
        feats = self.get_outputs(args, self.model)
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt

        return num_labels

    def get_outputs(self, args, model):
        
        dataloader = self.train_outputs['loader']

        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
       
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits = model(input_ids, segment_ids, input_mask, feature_ext = True)
                
                total_features = torch.cat((total_features, pooled_output))
               
        feats = total_features.cpu().numpy()

        return feats

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, data, 'bert', args.freeze_pretrain_bert_parameters)   
        self.optimizer , self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.pretrain_batch_size, \
            args.num_train_epochs, args.lr_pre, args.warmup_proportion)
        
        self.device = model.device
        
    def train(self, args, data):
        
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            contrast_dataloader = self.get_augment_dataloader(args, self.train_outputs)

            for batch in tqdm(contrast_dataloader, desc = "Iteration"):
                
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, _ = batch
                
                with torch.set_grad_enabled(True):
                    
                    input_ids_a, input_ids_b = self.batch_chunk(input_ids)
                    input_mask_a, input_mask_b = self.batch_chunk(input_mask)
                    segment_ids_a, segment_ids_b = self.batch_chunk(segment_ids)

                    aug_mlp_output_a, _ = self.model(input_ids_a, segment_ids_a, input_mask_a)
                    aug_mlp_output_b, _ = self.model(input_ids_b, segment_ids_b, input_mask_b)

                    norm_logits = F.normalize(aug_mlp_output_a)
                    norm_aug_logits = F.normalize(aug_mlp_output_b)

                    contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    
                    loss_contrast = self.contrast_criterion(contrastive_logits, temperature = args.pretrain_temperature, device = self.device)

                    loss = loss_contrast
                    
                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.zero_grad()
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                    
            loss = tr_loss / nb_tr_steps
            eval_results = {
                'train_loss': loss,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

        if args.save_model:
            pretrained_model_dir = os.path.join(args.method_output_dir, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)

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