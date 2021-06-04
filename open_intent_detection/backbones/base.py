import os 
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .PLM import BERT, Roberta, XLNet, freeze_parameters

backbones_map = {'bert':BERT, 'roberta':Roberta, 'xlnet': XLNet}


class ModelManager:

    def __init__(self, args, data):
        
        
        self.model, self.optimizer, self.scheduler = self.set_model(args, data)
        self.set_gpu(args, self.model)
    
    def set_model(self, args, data):

        backbone = backbones_map[args.backbone]

        if args.backbone == 'bert':
            model = backbone.from_pretrained('bert-base-uncased', num_labels=data.num_labels, cache_dir="cache")
        elif args.backbone == 'roberta':
            model = backbone.from_pretrained('roberta-base', num_labels=data.num_labels, cache_dir="cache")
        elif args.backbone == 'xlnet':
            model = backbone.from_pretrained('xlnet-base-cased', num_labels=data.num_labels, cache_dir="cache")
            
        if args.freeze_bert_parameters:
            model = freeze_parameters(model, args.backbone)
            
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(len(data.dataloader.train_labeled_examples) / args.train_batch_size) * args.num_train_epochs

        optimizer = AdamW(optimizer_grouped_parameters,
                        lr = args.lr,
                        correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args.num_warmup_steps, 
                                                    num_training_steps=args.num_training_steps)

        return model, optimizer, scheduler
    
    def set_gpu(self, args, model):

        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
        model.to(self.device)







