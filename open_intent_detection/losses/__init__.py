from .LMCL import LargeMarginCosineLoss
from .boundary import BoundaryLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'LargeMarginCosineLoss': LargeMarginCosineLoss(), 
                'BoundaryLoss': BoundaryLoss()
            }
