from .bert import BERT, BertForConstrainClustering, BertForDTC, BertForKCL_Similarity
from .glove import GloVeEmbeddingVectorizer
from .sae import get_stacked_autoencoder

backbones_map = {
                    'bert': BERT, 
                    'bert_CDAC': BertForConstrainClustering,
                    'bert_DTC': BertForDTC,
                    'bert_KCL_simi': BertForKCL_Similarity,
                    'glove': GloVeEmbeddingVectorizer,
                    'sae': get_stacked_autoencoder
                }