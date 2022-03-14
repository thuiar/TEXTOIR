from .ADB.manager import ADBManager
from .MSP.manager import MSPManager
from .DeepUnk.manager import DeepUnkManager
from .DOC.manager import DOCManager
from .OpenMax.manager import OpenMaxManager
from .MixUp.manager import MixUpManager
from .SEG.manager import SEGManager

method_map = {
                'ADB': ADBManager, 
                'DA-ADB': ADBManager, 
                'MSP': MSPManager, 
                'DeepUnk':DeepUnkManager, 
                'LOF': DeepUnkManager, 
                'DOC': DOCManager, 
                'OpenMax': OpenMaxManager, 
                'MixUp': MixUpManager,
                'SEG': SEGManager
            }
