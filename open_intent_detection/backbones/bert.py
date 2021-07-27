import torch
import math
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import L2_normalization

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class BERT(BertPreTrainedModel):

    def __init__(self, config, args, data):

        super(BERT, self).__init__(config)
        self.num_labels = data.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, data.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits


class BERT_Norm(BertPreTrainedModel):

    def __init__(self, config, args, data):

        super(BERT_Norm, self).__init__(config)
        self.num_labels = data.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        from torch.nn.utils import weight_norm
        self.norm = L2_normalization()
        self.classifier = weight_norm(
            nn.Linear(config.hidden_size, data.num_labels), name='weight')
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
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
