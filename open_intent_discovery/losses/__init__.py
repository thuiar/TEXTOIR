from torch import nn 
from .KCL import KCL
from .MCL import MCL

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'KCL': KCL(),
                'MCL': MCL()
            }
