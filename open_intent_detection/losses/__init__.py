from .CosineFaceLoss import CosineFaceLoss
from .CenterLoss import CenterLoss
from .ArcFaceLoss import ArcFaceLoss
from .SphereFaceLoss import SphereFaceLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'Binary_CrossEntropyLoss': nn.BCELoss(),
                'CosineFaceLoss': CosineFaceLoss(),
                'ArcFaceLoss': ArcFaceLoss(),
                'SphereFaceLoss': SphereFaceLoss(),
            }
