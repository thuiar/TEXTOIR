from .semi_supervised.DeepAligned.manager import DeepAlignedManager
from .semi_supervised.CDACPlus.manager import CDACPlusManager
from .semi_supervised.DTC_BERT.manager import DTCManager
from .semi_supervised.KCL_BERT.manager import KCLManager
from .unsupervised.KM.manager import KMManager
from .unsupervised.AG.manager import AGManager
from .unsupervised.SAE.manager import SAEManager
from .unsupervised.DEC.manager import DECManager
from .unsupervised.DCN.manager import DCNManager

method_map = {
                'DeepAligned': DeepAlignedManager, 
                'CDACPlus': CDACPlusManager,
                'DTC_BERT': DTCManager,
                'KCL_BERT': KCLManager,
                'KM': KMManager,
                'AG': AGManager,
                'SAE-KM': SAEManager,
                'DEC': DECManager,
                'DCN': DCNManager,
            }