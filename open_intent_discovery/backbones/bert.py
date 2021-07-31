from operator import mod
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class BERT(BertPreTrainedModel):
    
    def __init__(self,config, args):

        super(BERT, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)      
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
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

class BertForConstrainClustering(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForConstrainClustering, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        
        # train
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Pooling-mean
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.apply(self.init_bert_weights)
        
        # finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(args.num_labels, args.num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext = False, u_threshold=None, l_threshold=None, mode=None,  semi=False):

        eps = 1e-10
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1)) # Pooling-mean
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return logits
        else:
            if mode=='train':

                logits_norm = F.normalize(logits, p=2, dim=1)
                sim_mat = torch.matmul(logits_norm, logits_norm.transpose(0, -1))
                label_mat = labels.view(-1,1) - labels.view(1,-1)    
                label_mat[label_mat!=0] = -1 # dis-pair: label=-1
                label_mat[label_mat==0] = 1  # sim-pair: label=1
                label_mat[label_mat==-1] = 0 # dis-pair: label=0

                if not semi:
                    pos_mask = (label_mat > u_threshold).type(torch.cuda.FloatTensor)
                    neg_mask = (label_mat < l_threshold).type(torch.cuda.FloatTensor)
                    pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                    loss = (pos_entropy.mean() + neg_entropy.mean()) * 5

                    return loss

                else:
                    label_mat[labels==-1, :] = -1
                    label_mat[:, labels==-1] = -1
                    label_mat[label_mat==0] = 0
                    label_mat[label_mat==1] = 1
                    pos_mask = (sim_mat > u_threshold).type(torch.cuda.FloatTensor)
                    neg_mask = (sim_mat < l_threshold).type(torch.cuda.FloatTensor)
                    pos_mask[label_mat==1] = 1
                    neg_mask[label_mat==0] = 1
                    pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                    loss = pos_entropy.mean() + neg_entropy.mean() + u_threshold - l_threshold

                    return loss

            else:
                q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
                q = q.pow((self.alpha + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, 1)).t() # Make sure each sample's n_values add up to 1.
                return logits, q

class BertForDTC(BertPreTrainedModel):
    def __init__(self, config, args):

        super(BertForDTC, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)

        #train
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = loss_fct(logits, labels)
            return loss
        else:
            return pooled_output, logits