import torch
import math
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import L2_normalization

class BERT(BertPreTrainedModel):
    
    def __init__(self,config, num_labels):

        super(BERT, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)      
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
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

class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

class BERT_Disaware(BertPreTrainedModel):
    
    def __init__(self,config, num_labels):
    
        super(BERT_Disaware, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cosnorm_classifier = CosNorm_Classifier(config.hidden_size, num_labels)
        
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None, centroids = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        x = pooled_output

        if feature_ext:
            return pooled_output

        else:

            direct_feature = x
            batch_size = x.size(0)
            feat_size = x.size(1)

            #set up visual memory
            x_expand = x.unsqueeze(1).expand(-1, self.num_labels, -1)
            centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
            keys_memory = centroids

            #computing reachability
            dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
            values_nn, labels_nn = torch.sort(dist_cur, 1)
            
            dis = values_nn[:, 0]
            scale = 1.0
            reachability = (scale / dis).unsqueeze(1).expand(-1, feat_size)
            x = reachability * direct_feature
            logits = self.cosnorm_classifier(x)

            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits