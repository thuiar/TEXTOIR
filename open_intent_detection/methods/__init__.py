from .ADB.manager import ADBManager
from .ADBdisaware.manager import ADBdisawareManager
from .MSP.manager import MSPManager
from .DeepUnk.manager import DeepUnkManager
from .DOC.manager import DOCManager
from .OpenMax.manager import OpenMaxManager
from .MixUp.manager import MixUpManager
from .SEG.manager import SEGManager

method_map = {
                'ADB': ADBManager, 
                'ADBdisaware': ADBdisawareManager,
                'MSP': MSPManager, 
                'DeepUnk':DeepUnkManager, 
                'DOC': DOCManager, 
                'OpenMax': OpenMaxManager, 
                'MixUp': MixUpManager,
                'SEG': SEGManager
            }
