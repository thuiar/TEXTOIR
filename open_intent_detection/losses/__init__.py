from .CosineFaceLoss import CosineFaceLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'Binary_CrossEntropyLoss': nn.BCELoss(),
                'CosineFaceLoss': CosineFaceLoss()
            }
