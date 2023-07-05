import os 
import torch
import math
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from .utils import freeze_bert_parameters, set_allow_growth
from .__init__ import backbones_map


class ModelManager:

    def __init__(self, args, data, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        
        if args.method in ['KM', 'AG', 'SAE', 'DEC', 'DCN']:
            set_allow_growth('0')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    
    def set_optimizer(self, model, num_train_examples, train_batch_size, num_train_epochs, lr, warmup_proportion):

        num_train_optimization_steps = int(num_train_examples / train_batch_size) * num_train_epochs
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = lr, correct_bias=False)
        num_warmup_steps= int(num_train_examples * num_train_epochs * warmup_proportion / train_batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        return optimizer ,scheduler
    
    def set_model(self, args, data, pattern, freeze_parameters = True):
        
        backbone = backbones_map[args.backbone]

        if pattern == 'bert':
            
            if hasattr(backbone, 'from_pretrained'):
                model = backbone.from_pretrained(args.pretrained_bert_model, args = args)  
            else:
                model = backbone(args)
                
            if freeze_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                model = freeze_bert_parameters(model)
            
            model.to(self.device)
            
            return model

        elif args.setting == 'unsupervised':

            if pattern == 'glove':

                self.logger.info("Building GloVe (D=300)...")
                
                gev = backbone(data.dataloader.embedding_matrix, data.dataloader.index_word, data.dataloader.train_data)
                emb_train = gev.transform(data.dataloader.train_data, method='mean')
                emb_test = gev.transform(data.dataloader.test_data, method='mean')
                
                self.logger.info('Building finished!')

                return emb_train, emb_test
        
            elif pattern == 'sae':

                self.logger.info("Building TF-IDF Vectors...")
                sae = backbone(data.dataloader.tfidf_train.shape[1])

                return sae
    


        









