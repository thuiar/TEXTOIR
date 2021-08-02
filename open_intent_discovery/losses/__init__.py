from torch import nn 
from .KCL import KCL

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'KCL': KCL()
            }
