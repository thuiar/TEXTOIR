import numpy as np
import os   
import random
import torch
import logging

from .__init__ import max_seq_lengths, backbone_loader_map, benchmark_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class DataManager:
    
    def __init__(self, args, logger_name = 'Detection'):
        
        self.logger = logging.getLogger(logger_name)

        set_seed(args.seed)
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.all_label_list = self.get_labels(args.dataset)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False)
        self.known_label_list = list(self.known_label_list)

        self.logger.info('The number of known intents is %s', self.n_known_cls)
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))

        args.num_labels = self.num_labels = len(self.known_label_list)

        if args.dataset == 'oos':
            self.unseen_label = 'oos'
        else:
            self.unseen_label = '<UNK>'
        
        args.unseen_label_id = self.unseen_label_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_label]

        self.anum_labels = args.anum_labels = len(self.label_list)
        self.dataloader = self.get_loader(args, self.get_attrs())

    def get_labels(self, dataset):
        
        labels = benchmark_labels[dataset]

        return labels
    
    def get_loader(self, args, attrs):
        
        dataloader = backbone_loader_map[args.backbone](args, attrs, args.logger_name)

        return dataloader
    
    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
