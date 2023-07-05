import logging
import numpy as np
import copy
import torch
import torch.nn as nn

from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from sklearn.metrics import silhouette_score
from utils.functions import set_seed
from utils.functions import save_model
from sentence_transformers import SentenceTransformer

class SCCLmanager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        loader = data.dataloader
        self.num_labels = data.num_labels
        self.train_dataloader, self.test_dataloader = \
            loader.train_outputs['loader'], loader.test_outputs['loader']
        self.train_input_ids, self.train_input_mask, self.train_segment_ids = \
            loader.train_outputs['input_ids'], loader.train_outputs['input_mask'], loader.train_outputs['segment_ids']
        
        self.augdataloader = self.get_augment_dataloader(args)
        self.tokenizer = loader.tokenizer
        self.set_model_optimizer(args, data, model)
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=args.temperature)

    def set_model_optimizer(self, args, data, model):
            
        self.model = model.set_model(args, data, 'bert', args.freeze_bert_parameters)   
        self.model.bert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')[0].auto_model
        self.device = model.device 
        self.model.bert.to(self.device)

        cluster_centers = self.get_kmeans_centers(self.train_dataloader, args)
   
        self.model.init_model(cluster_centers=cluster_centers, alpha=args.alpha) 
        self.optimizer = self.get_optimizer(self.model, args)
        self.model.to(self.device)


    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()

    def get_optimizer(self, model, args): 

        optimizer = torch.optim.Adam([
            {'params':model.bert.parameters()}, 
            {'params':model.contrast_head.parameters(), 'lr': args.lr * args.lr_scale},
            {'params':model.cluster_centers, 'lr': args.lr * args.lr_scale}
        ], lr = args.lr)
        
        return optimizer 

    def get_kmeans_centers(self, train_loader, args):
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(self.device) for t in batch)
            train_input_ids, train_input_mask, _, _= batch
            corpus_embeddings = self.model.get_mean_embeddings(train_input_ids, train_input_mask)
            if i == 0:     
                all_embeddings = corpus_embeddings.cpu().detach().numpy()
            else:
                all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)

        print('embedding shape', all_embeddings.shape)
        clustering_model = KMeans(n_clusters=self.num_labels, random_state=args.seed)
        clustering_model.fit(all_embeddings)

        print("Iterations:{},  centers:{}".format(clustering_model.n_iter_,   clustering_model.cluster_centers_.shape))
        
        return clustering_model.cluster_centers_
    
    def train(self, args, data):

        self.logger.info('SCCL training starts...')
 
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            self.model.train()
            
            tr_loss, nb_tr_steps = 0, 0
            for batch in tqdm(self.augdataloader, desc="Training(All)"):
                with torch.set_grad_enabled(True):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids = batch
                    embd1, embd2, embd3 = self.model(input_ids, input_mask, task_type="explicit")
                    # Instance-CL loss
                    feat1, feat2 = self.model.contrast_logits(embd2, embd3)
                    losses = self.contrast_loss(feat1, feat2)
                    loss = args.eta * losses["loss"]

                    output = self.model.get_cluster_prob(embd1)
                    target = self.target_distribution(output).detach()
                    cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
                    loss += cluster_loss
                    losses["cluster_loss"] = cluster_loss.item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tr_loss += loss.item()
                    nb_tr_steps += 1

            train_loss = tr_loss / nb_tr_steps
            self.logger.info("***** Epoch: %s: train results *****", str(epoch))
            self.logger.info("  train_loss = %s",  str(train_loss))

        self.logger.info('SCCL training finished...')

        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):

        feats, y_true = self.get_outputs(args, mode = 'test', model = self.model)
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
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
        
        if mode == 'eval':
            dataloader = self.train_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.augdataloader
            
        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output = model(input_ids, input_mask, task_type = 'evaluate')
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))

     
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        return feats, y_true

    def get_augment_dataloader(self, args):

        train_input_ids = self.train_input_ids.unsqueeze(1)
        train_input_mask = self.train_input_mask.unsqueeze(1)
        train_segment_ids = self.train_segment_ids.unsqueeze(1)
        
        train_input_ids = torch.cat(([train_input_ids, train_input_ids, train_input_ids]), dim = 1)
        train_input_mask = torch.cat(([train_input_mask, train_input_mask, train_input_mask]), dim = 1)
        train_segment_ids = torch.cat(([train_segment_ids, train_segment_ids, train_segment_ids]), dim = 1)

        train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler,  batch_size = args.train_batch_size )

        return train_dataloader


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}
            



