from .ADB.manager import ADBManager
from .MSP.manager import MSPManager
from .DeepUnk.manager import DeepUnkManager
from .DOC.manager import DOCManager
from .OpenMax.manager import OpenMaxManager

method_map = {
                'ADB': ADBManager, 
                'MSP': MSPManager, 
                'DeepUnk':DeepUnkManager, 
                'DOC': DOCManager, 
                'OpenMax': OpenMaxManager, 
            }