from .LMCosine import LargeMarginCosineLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'LargeMarginCosineLoss': LargeMarginCosineLoss(), 
            }
