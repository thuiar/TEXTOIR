import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import ConvexSampler

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class BERT_DOC(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_DOC, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids = None):
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = encoded_layer_12[-1].mean(dim=1)
        
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.activation(pooled_output)
        
        logits = self.classifier(pooled_output)
        logits = self.dropout(logits)
        sigmoid = nn.Sigmoid()
        logits = sigmoid(logits)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                target = F.one_hot(labels, num_classes = self.num_labels)
                loss_bce = loss_fct(logits, target.float())
                return loss_bce
            else:
                return pooled_output, logits

class BERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids = None):
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = encoded_layer_12[-1].mean(dim=1)
        
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss_ce = loss_fct(logits, labels)
                return loss_ce
            else:
                return pooled_output, logits

class BERT_Norm(BertPreTrainedModel):
    def __init__(self, config, args):

        super(BERT_Norm, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.weight = Parameter(torch.FloatTensor(args.num_labels, args.feat_dim).to(args.device))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, device = None, head = None):
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = encoded_layer_12[-1].mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.normalize(pooled_output)

        logits = F.linear(pooled_output, F.normalize(self.weight))
        logits = F.softmax(logits, dim = 1)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits

class BERT_MixUp(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_MixUp, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sampler = ConvexSampler(args)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels + 1)
        self.t = args.temp
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        if mode is not 'test':
            pooled_output, labels = self.sampler(pooled_output, labels, mode=mode)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(torch.div(logits, self.t), labels)
                return loss
            else:
                return pooled_output, logits, labels

class BERT_SEG(BertPreTrainedModel):
    def __init__(self, config, args):
        
        super(BERT_SEG, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.alpha = args.alpha
        self.lambda_ = args.lambda_
        self.means = nn.Parameter(torch.randn(self.num_labels, args.feat_dim).cuda())
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))


    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, device=None, p_y = None, class_emb=None, loss_fct=None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            
            batch_size = pooled_output.shape[0]

            XY = torch.matmul(pooled_output, torch.transpose(self.means, 0, 1))
            XX = torch.sum(pooled_output ** 2, dim=1, keepdim=True)
            YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
            neg_sqr_dist = - 0.5 * (XX - 2.0 * XY + YY)
            
            # with p_y
            ########################################
            p_y = p_y.expand_as(neg_sqr_dist).to(device)  # [bsz, n_c_seen]
            # torch.exp(x) e^x
            dist_exp = torch.exp(neg_sqr_dist)
            dist_exp_py = p_y.mul(dist_exp)
            dist_exp_sum = torch.sum(dist_exp_py, dim=1, keepdim=True)  # [bsz, n_c_seen] -> [bsz, 1]
            logits = dist_exp_py / dist_exp_sum  # [bsz, n_c, seen]

            if mode == 'train':
                
                labels_reshped = labels.view(labels.size()[0], -1)  # [bsz] -> [bsz, 1]
                ALPHA = torch.zeros(batch_size, self.num_labels).to(device).scatter_(1, labels_reshped, self.alpha)  # margin
                K = ALPHA + torch.ones([batch_size, self.num_labels]).to(device)

                #######################################
                dist_margin = torch.mul(neg_sqr_dist, K)
                dist_margin_exp = torch.exp(dist_margin)
                dist_margin_exp_py = p_y.mul(dist_margin_exp)
                dist_exp_sum_margin = torch.sum(dist_margin_exp_py, dim=1, keepdim=True)
                likelihood = dist_margin_exp_py / dist_exp_sum_margin
                loss_ce = - likelihood.log().sum() / batch_size
                
                #######################################
                means = self.means if class_emb is None else class_emb
                means_batch = torch.index_select(means, dim=0, index=labels)
                loss_gen = (torch.sum((pooled_output - means_batch)**2) / 2) * (1. / batch_size)
                ########################################
                loss = loss_ce + self.lambda_ * loss_gen
                return loss

            else:
                return pooled_output, logits

class CosNorm_Classifier(nn.Module):

    def __init__(self, in_dims, out_dims, scale=64, device = None):

        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).to(device))
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

    def __init__(self, config, args):

        super(BERT_Disaware, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.cosnorm_classifier = CosNorm_Classifier(
            config.hidden_size, args.num_labels, args.scale, args.device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, dist_infos = None):

        outputs = self.bert(
            input_ids, token_type_ids, attention_mask,  output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        x = pooled_output

        if feature_ext:
            return pooled_output

        else:

            feat_size = x.shape[1]
            batch_size = x.shape[0]

            f_expand = x.unsqueeze(1).expand(-1, self.num_labels, -1)
            centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)        
            dist_cur = torch.norm(f_expand - centroids_expand, 2, 2)
            values_nn, labels_nn = torch.sort(dist_cur, 1)        

            nearest_centers = centroids[labels_nn[:, 0]]
            dist_denominator = torch.norm(x - nearest_centers, 2, 1)
            second_nearest_centers = centroids[labels_nn[:, 1]]
            dist_numerator = torch.norm(x - second_nearest_centers, 2, 1)
            
            dist_info = dist_numerator - dist_denominator
            dist_info = torch.exp(dist_info)
            scalar = dist_info

            reachability = scalar.unsqueeze(1).expand(-1, feat_size)
            x = reachability * pooled_output

            logits = self.cosnorm_classifier(x)

            if mode == 'train':

                loss = loss_fct(logits, labels)

                return loss

            elif mode == 'eval':

                return pooled_output, logits
