import numpy as np
import os   
import random
import torch
from .bert_loader import BERT_Loader
from .unsup_loader import UNSUP_Loader

max_seq_lengths = {
                        'clinc':30, 
                        'banking':55, 
                    }
                    
backbone_loader_map = {
                            'bert': BERT_Loader,
                            'glove': UNSUP_Loader,
                            'sae': UNSUP_Loader
                      }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DataManager:
    
    def __init__(self, args):

        set_seed(args.seed)
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = self.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        self.dataloader = self.get_loader(args, self.get_attrs())

                
    def get_labels(self, data_dir):
        
        labels = np.load(os.path.join(data_dir, 'labels.npy'), allow_pickle=True)

        return labels
    
    def get_loader(self, args, attrs):
        
        dataloader = backbone_loader_map[args.backbone](args, attrs)

        return dataloader
    
    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs



