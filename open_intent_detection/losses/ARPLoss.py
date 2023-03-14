import torch
import torch.nn as nn
import torch.nn.functional as F
from .Dist import Dist

class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, args):
        super(ARPLoss, self).__init__()
        self.weight_pl = float(args.weight_pl)
        self.device = args.device
        self.temp = args.temp
        self.Dist = Dist(num_classes=args.num_labels, num_centers=1, feat_dim=args.feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, labels=None, center=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        # dist_dot_p = self.Dist(x, center=center, metric='dot')
        # dist_l2_p = self.Dist(x, center=center)
        logits = dist_l2_p - dist_dot_p

        if labels is None: return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        # center_batch = center[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).to(self.device)
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

# class ARPLoss(nn.CrossEntropyLoss):
#     def __init__(self, args):
#         super(ARPLoss, self).__init__()
#         self.weight_pl = float(args.weight_pl)
#         self.temp = args.temp
#         self.Dist = Dist(num_classes=args.num_labels, num_centers=1, feat_dim=args.feat_dim)
#         self.radius = 1
#         self.radius = nn.Parameter(torch.Tensor(self.radius))
#         self.radius.data.fill_(0)

#     def forward(self, x, labels=None):
#         dist = self.Dist(x)
#         logits = F.softmax(dist, dim=1)
#         if labels is None: return logits, 0
#         loss = F.cross_entropy(dist / self.temp, labels)
#         center_batch = self.Dist.centers[labels, :]
#         _dis = (x - center_batch).pow(2).mean(1)
#         loss_r = F.mse_loss(_dis, self.radius)
#         loss = loss + self.weight_pl * loss_r

#         return logits, loss


