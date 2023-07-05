from torch import nn 
from .KCL import KCL
from .MCL import MCL
from .SupConLoss import SupConLoss

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'KCL': KCL(),
                'MCL': MCL(),
                'SupConLoss': SupConLoss()
            }
