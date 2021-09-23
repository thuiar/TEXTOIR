import torch
import math
import torch.nn.functional as F
from torch import nn

class SphereFaceLoss(nn.Module):

    """
    cos_theta and target need to be normalized first
    """ 

    def __init__(self, m=1.35, s=64):
        
        super(SphereFaceLoss, self).__init__()
        self.m = m
        self.s = s

    def forward(self, cos_theta, target):
        
        phi_theta = torch.cos(torch.acos(cos_theta) * self.m) 

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)
        
        return F.cross_entropy(self.s * output, target)