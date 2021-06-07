from .LMCosine import LargeMarginCosineLoss
from .CenterLoss import CenterLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'LargeMarginCosineLoss': LargeMarginCosineLoss(), 
                'center_loss': CenterLoss()
            }
