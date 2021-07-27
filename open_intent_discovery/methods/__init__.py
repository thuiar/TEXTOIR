from .semi_supervised.DeepAligned.manager import DeepAlignedManager
from .unsupervised.KM.manager import KMManager
from .unsupervised.AG.manager import AGManager
from .unsupervised.SAE.manager import SAEManager

method_map = {
                'DeepAligned': DeepAlignedManager, 
                'KM': KMManager,
                'AG': AGManager,
                'SAE-KM': SAEManager,
            }