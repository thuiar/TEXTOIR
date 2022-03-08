from .bert import BERT, BERT_Norm, BERT_MixUp, BERT_SEG, BERT_Disaware, BERT_DOC

backbones_map = {
                    'bert': BERT, 
                    'bert_norm': BERT_Norm,
                    'bert_mixup': BERT_MixUp,
                    'bert_seg': BERT_SEG,
                    'bert_disaware': BERT_Disaware,
                    'bert_doc': BERT_DOC,
                }