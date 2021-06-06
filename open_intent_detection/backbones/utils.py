import torch
from torch import nn

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
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

