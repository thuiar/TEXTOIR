from .LMCL import LargeMarginCosineLoss
from .boundary import BoundaryLoss
from torch import nn 

loss_map = {'cross_entropy': nn.CrossEntropyLoss(), 'LMCL': LargeMarginCosineLoss(), 'boundary': BoundaryLoss()}