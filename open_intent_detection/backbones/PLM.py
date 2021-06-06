import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from .utils import L2_normalization

class BERT(BertPreTrainedModel):
    
    def __init__(self,config, num_labels, classifier_norm = False):

        super(BERT, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if classifier_norm:
            self.classifier = weight_norm(nn.Linear(config.hidden_size, num_labels), name='weight')
        else:
            self.classifier = nn.Linear(config.hidden_size,num_labels)
        
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None, feat_norm = False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        if feat_norm:
            pooled_output = self.norm(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits

class BERT_DeepUnk(BertPreTrainedModel):

    def __init__(self,config, num_labels):
    
        super(BERT_DeepUnk, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        from torch.nn.utils import weight_norm
        self.classifier = weight_norm(nn.Linear(config.hidden_size, num_labels), name='weight')
        self.norm = L2_normalization()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits
# class BERT_LargeMarginCosineLoss(BertPreTrainedModel):

#     def __init__(self,config, num_labels):
    
#         super(BERT, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         if args.loss == 'LargeMarginCosineLoss':
#             self.norm = L2_normalization()
#             self.classifier = weight_norm(nn.Linear(config.hidden_size, num_labels), name='weight')

#         else:
#             self.classifier = nn.Linear(config.hidden_size,num_labels)
        
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
#                 feature_ext = False, mode = None, loss_fct = None):

#         encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
#         pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
#         pooled_output = self.activation(pooled_output)
#         pooled_output = self.dropout(pooled_output)
        
#         if loss_fct == 'LargeMarginCosineLoss':
#             pooled_output = self.norm(pooled_output)

#         logits = self.classifier(pooled_output)

#         if feature_ext:
#             return pooled_output
#         else:
#             if mode == 'train':
#                 loss = loss_fct(logits, labels)
#                 return loss
#             else:
#                 return pooled_output, logits