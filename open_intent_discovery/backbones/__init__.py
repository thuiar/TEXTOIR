from .bert import BertForConstrainClustering, BertForDTC, BertForKCL_Similarity, \
    BertForKCL, BertForMCL, BERT_MTP_Pretrain, BERT_MTP, Bert_SCCL, BERT_GCD, BERT_CC, BERTForDeepAligned, \
        BERT_USNID, BERT_USNID_UNSUP
from .glove import GloVeEmbeddingVectorizer
from .sae import get_stacked_autoencoder

backbones_map = {   
                    'bert_CDAC': BertForConstrainClustering,
                    'bert_DTC': BertForDTC,
                    'bert_KCL_simi': BertForKCL_Similarity,
                    'bert_KCL': BertForKCL,
                    'bert_MCL': BertForMCL,
                    'bert_MTP_Pretrain': BERT_MTP_Pretrain,
                    'bert_MTP' : BERT_MTP,
                    'bert_USNID': BERT_USNID,
                    'bert_USNID_Unsup': BERT_USNID_UNSUP,
                    'glove': GloVeEmbeddingVectorizer,
                    'sae': get_stacked_autoencoder,
                    'bert_SCCL' : Bert_SCCL,
                    'bert_GCD' : BERT_GCD,
                    'bert_CC' : BERT_CC,
                    'bert_DeepAligned' : BERTForDeepAligned
                }