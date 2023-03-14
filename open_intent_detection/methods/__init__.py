from .ADB.manager import ADBManager
from .MSP.manager import MSPManager
from .DeepUnk.manager import DeepUnkManager
from .DOC.manager import DOCManager
from .OpenMax.manager import OpenMaxManager
from .K_1_way.manager import K_1_wayManager
from .SEG.manager import SEGManager
from .MDF.manager import MDFManager
from .ARPL.manager import ARPLManager

method_map = {
                'ADB': ADBManager, 
                'DA-ADB': ADBManager, 
                'MSP': MSPManager, 
                'DeepUnk':DeepUnkManager, 
                'LOF': DeepUnkManager, 
                'DOC': DOCManager, 
                'OpenMax': OpenMaxManager, 
                'K+1-way': K_1_wayManager,
                'SEG': SEGManager,
                'MDF': MDFManager,
                'ARPL': ARPLManager
            }
