import torch
from torch import nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Deep Open Intent Classification with Adaptive Decision Boundary.
    https://arxiv.org/pdf/2012.10209.pdf
    """
    def __init__(self, num_labels=10, feat_dim=2, device = None):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).to(device))
        nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels, w = 1):
        
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        eps = 1e-10

        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        
        # loss = pos_loss.sum() / (pos_mask.sum() + eps) + neg_loss.sum() / (neg_mask.sum() + eps)
        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta 