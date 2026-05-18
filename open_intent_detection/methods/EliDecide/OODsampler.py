import torch
import numpy as np
import random
from torch import nn
from scipy.spatial.distance import cdist

def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

class OODSampler(nn.Module):

    def __init__(self, args):
        super(OODSampler, self).__init__()
        self.args = args

    def alternate_mixup(self, data1, data2):
        mixed_data = torch.zeros_like(data1)
        for i in range(data1.size(0)):
            if i % 2 == 0:
                mixed_data[i, :] = data1[i, :]
            else:
                mixed_data[i, :] = data2[i, :]
        return mixed_data

    def forward(self, ind, labels, num_ood):

        ood_list = []

        label_set = set(labels.tolist())

        
        while len(ood_list) < num_ood:
            
            if self.args.select_number_min == self.args.select_number_max:
                select_number = self.args.select_number_min
            else:    
                select_number = np.random.randint(self.args.select_number_min, self.args.select_number_max + 1)

            if select_number > len(label_set):
                return None
            select_label = np.random.choice(list(label_set), select_number, replace=False)
            cdt = []
            for label in select_label:
                idx = torch.where(labels == label)[0].cpu()
                cdt.append(np.random.choice(idx, 1))
            s = np.random.dirichlet(alpha=[self.args.alpha] * select_number)
            ood = sum(s[i] * ind[cdt[i]] for i in range(select_number)).view(-1)
            ood_list.append(ood)
    
        return ood_list