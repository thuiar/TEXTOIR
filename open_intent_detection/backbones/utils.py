import torch
from torch import nn
import numpy as np

def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class L2_normalization(nn.Module):
    def forward(self, input):
        return l2_norm(input)   

def freeze_bert_parameters(model):
    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model

def freeze_bert_parameters_KCL(model):
    for name, param in model.encoder_q.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    for name, param in model.encoder_k.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model

class ConvexSampler(nn.Module):
    def __init__(self, args):
        super(ConvexSampler, self).__init__()
        self.multiple_convex = args.multiple_convex
        self.multiple_convex_eval = args.multiple_convex_eval
        self.unseen_label_id = args.unseen_label_id
        self.device = args.device
        self.batch_size = args.train_batch_size
        self.oos_num = args.train_batch_size
        self.feat_dim = args.feat_dim

    def forward(self, z, label_ids, mode=None):
        num_convex = self.batch_size * self.multiple_convex
        num_convex_eval = self.batch_size * self.multiple_convex_eval
        convex_list = []
        if mode =='train':
            if label_ids.size(0)>2:
                while len(convex_list) < num_convex:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    # cdt = np.random.choice(label_ids.size(0) - self.oos_num, 2, replace=False)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(num_convex, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.unseen_label_id] * num_convex).to(self.device)), dim=0)
        elif mode == 'eval':
            if label_ids.size(0) > 2:
                val_num = num_convex_eval
                while len(convex_list) < val_num:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(val_num, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.unseen_label_id] * val_num).to(self.device)), dim=0)
        return z, label_ids

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.1):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()
    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()
