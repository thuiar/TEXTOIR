from .bert import BERT, BertForConstrainClustering, BertForDTC, BertForKCL_Similarity, BertForKCL
from .glove import GloVeEmbeddingVectorizer
from .sae import get_stacked_autoencoder

backbones_map = {
                    'bert': BERT, 
                    'bert_CDAC': BertForConstrainClustering,
                    'bert_DTC': BertForDTC,
                    'bert_KCL_simi': BertForKCL_Similarity,
                    'bert_KCL': BertForKCL,
                    'glove': GloVeEmbeddingVectorizer,
                    'sae': get_stacked_autoencoder
                }