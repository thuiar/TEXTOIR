import torch
import tensorflow as tf
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

def set_allow_growth(device):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    config.gpu_options.visible_device_list = device
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess) 

def PairEnum(x,mask=None):

    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))

    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))

        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2