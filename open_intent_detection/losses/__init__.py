from .CosineFaceLoss import CosineFaceLoss
from .CenterLoss import CenterLoss
from .ArcFaceLoss import ArcFaceLoss
from .SphereFaceLoss import SphereFaceLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'CosineFaceLoss': CosineFaceLoss(),
                'CenterLoss': CenterLoss(),
                'ArcFaceLoss': ArcFaceLoss(),
                'SphereFaceLoss': SphereFaceLoss()
            }
