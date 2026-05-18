import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
from .OODsampler import OODSampler

import torch.types
#import sys


class BoundaryLoss(nn.Module):
    def __init__(self, args, device=None):
        super(BoundaryLoss, self).__init__()
        self.num_labels = args.num_labels
        self.beta = args.beta
        self.feat_dim = args.feat_dim
        self.device = device
        self.args = args
        if 'shape' not in self.args:
            self.args.triangle = False
        if 'shape' not in self.args:
            self.args.shape = "ellipsoid"
        if 'ood' not in self.args:
            self.args.ood = True
        self.L = nn.Parameter(torch.zeros((self.num_labels, self.feat_dim * (self.feat_dim - 1) // 2), device=self.device))
        if not self.args.triangle:
            self.U = nn.Parameter(torch.zeros((self.num_labels, self.feat_dim * (self.feat_dim - 1) // 2), device=self.device))
        self.D = nn.Parameter(torch.ones((self.num_labels, self.feat_dim), device=self.device))
        self.dropout = torch.nn.Dropout(0.5)
        # for i in range(num_labels):
        #     for j in range(self.feat_dim):
        #         self.L[i, (self.feat_dim * 2 - (j - 1)) * j // 2] = 0.5
        self.C = torch.zeros((self.num_labels, self.feat_dim)).to(self.device)
        self.OODSampler = OODSampler(args)

        self.count = 20

    def get_rotate_matrix(self):
        rotate_matrix = torch.zeros((self.num_labels, self.feat_dim, self.feat_dim)).to(self.device)

        indices = torch.tril_indices(self.feat_dim, self.feat_dim, offset=-1)
        dia_indices = range(0, self.feat_dim)
        # L = self.dropout(self.L)
        for i in range(self.num_labels):
            if self.args.shape == "ball":
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, 0]
            elif self.args.shape == "regular_ellipsoid":
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, dia_indices]
            else:
                rotate_matrix[i, indices[0], indices[1]] = self.L[i]
                if not self.args.triangle:
                    rotate_matrix[i, indices[1], indices[0]] = self.U[i]
                else:
                    rotate_matrix[i, indices[1], indices[0]] = self.L[i]
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, dia_indices]
        # if not self.triangle:
        #     rotate_matrix = self.dropout(rotate_matrix)
        return rotate_matrix

    def forward(self, pooled_output, centroids, delta, labels, get_rotate_x=False):

        rotate_matrix = self.get_rotate_matrix()

        if self.args.ood:
            ood = self.OODSampler(pooled_output, labels, pooled_output.shape[0])
            if ood != None:
                ood = torch.stack(ood, dim=0)
            
        d = delta[labels]
        c = centroids[labels]
        x = pooled_output - c

        rotate_x = torch.bmm(rotate_matrix[labels], x.unsqueeze(2)).squeeze(2)
        
        if get_rotate_x:
            return rotate_x

        euc_dis = torch.norm(rotate_x, 2, 1)# / d
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        # pos_loss = (1 - torch.exp(1 - euc_dis)) * pos_mask
        # pos_loss = torch.log(euc_dis / d) * pos_mask
        # print(euc_dis)
        # pos_loss = (euc_dis - d) * pos_mask

        in_mask = (d > euc_dis)
        out_mask = ~in_mask
        pos_loss = (euc_dis - d) * out_mask
        
        pos_num = pos_mask.sum()
        
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        neg_num = (neg_mask).sum()
        
        if not self.args.ood:
            neg_loss = (d - euc_dis) * neg_mask * self.num_labels
            # neg_loss = torch.exp(euc_dis - d) * neg_mask
        else:
            neg_loss = torch.zeros((ood.shape[0] if ood is not None else 1, )).to(self.device)
            if ood != None:
                for k in range(self.num_labels):
                    # neg_mask = (labels != k)
                    rotate_x = torch.mm(rotate_matrix[k], (ood - centroids[k]).transpose(-1, -2)).transpose(-1, -2)
                    euc_dis = torch.norm(rotate_x, 2, 1)
                    in_mask = (delta[k] > euc_dis)
                    out_mask = ~in_mask
                    # if self.count == 1:
                    #     print(in_mask.sum(), out_mask.sum(), euc_dis.mean())
                    neg_loss += ((delta[k] - euc_dis + self.beta) * in_mask + self.beta * torch.exp(delta[k] - euc_dis) * out_mask)
        # self.count -= 1
        # if self.count == 0:
        #     exit(0)
        
        eps = 1e-6
        loss = pos_loss.mean() + neg_loss.mean()

        # if self.count == 0:
        #     exit(0)
        # else:
        #     self.count -= 1

        return pos_loss.mean(), neg_loss.mean(), pos_num, neg_num, loss
        