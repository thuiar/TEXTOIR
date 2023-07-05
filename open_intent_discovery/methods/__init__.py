from .semi_supervised.USNID.manager import USNIDManager
from .semi_supervised.DeepAligned.manager import DeepAlignedManager
from .semi_supervised.CDACPlus.manager import CDACPlusManager
from .semi_supervised.DTC_BERT.manager import DTCManager
from .semi_supervised.KCL_BERT.manager import KCLManager
from .semi_supervised.MCL_BERT.manager import MCLManager
from .semi_supervised.GCD.manager import GCDManager
from .semi_supervised.MTP_CLNN.manager import MTP_CLNNManager
from .unsupervised.KM.manager import KMManager
from .unsupervised.AG.manager import AGManager
from .unsupervised.SAE.manager import SAEManager
from .unsupervised.DEC.manager import DECManager
from .unsupervised.DCN.manager import DCNManager
from .unsupervised.SCCL.manager import SCCLmanager
from .unsupervised.CC.manager import CCmanager
from .unsupervised.USNID.manager import UnsupUSNIDManager


method_map = {
                'USNID': USNIDManager, 
                'DeepAligned': DeepAlignedManager, 
                'CDACPlus': CDACPlusManager,
                'DTC_BERT': DTCManager,
                'KCL_BERT': KCLManager,
                'MCL_BERT': MCLManager,
                'KM': KMManager,
                'AG': AGManager,
                'SAE': SAEManager,
                'DEC': DECManager,
                'DCN': DCNManager,
                'GCD' : GCDManager,
                'MTP_CLNN':MTP_CLNNManager,
                'SCCL' : SCCLmanager,
                'CC' : CCmanager,
                'UnsupUSNID': UnsupUSNIDManager
            }