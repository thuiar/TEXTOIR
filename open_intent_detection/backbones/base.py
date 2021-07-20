import os 
import torch
import math
import logging
from pytorch_pretrained_bert.optimization import BertAdam
from .bert import BERT, BERT_DeepUnk
from .utils import freeze_bert_parameters

backbones_map = {
                    'bert': BERT, 
                    'bert_deepunk': BERT_DeepUnk,
                }

class ModelManager:

    def __init__(self, args, data, logger_name):

        self.logger = logging.getLogger(logger_name)
        
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   

        self.model, self.optimizer = self.set_model(args, data)
    
    def set_model(self, args, data):
        
        backbone = backbones_map[args.backbone]

        if args.backbone[:4] == 'bert':

            model = backbone.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)    
            model.to(self.device)

            if args.freeze_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                model = freeze_bert_parameters(model)
            
            
            num_train_optimization_steps = int(len(data.dataloader.train_labeled_examples) / args.train_batch_size) * args.num_train_epochs
            
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            self.logger.info('The number of training optimization steps are %s', num_train_optimization_steps)

            optimizer = BertAdam(optimizer_grouped_parameters,
                            lr = args.lr,
                            warmup = args.warmup_proportion,
                            t_total = num_train_optimization_steps)  

        return model, optimizer









