import torch
import math
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.parameter import Parameter
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

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

class BERT_K_1_way(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_K_1_way, self).__init__(config)
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

class BERT_MDF_Pretrain(nn.Module):
    
    def __init__(self, args):

        super(BERT_MDF_Pretrain, self).__init__()
        self.num_labels = args.num_labels
        self.bert = BertForMaskedLM.from_pretrained(args.pretrained_bert_model)
        self.dropout = nn.Dropout(0.1) #0.1
        self.classifier = nn.Linear(args.feat_dim, args.num_labels)  
        
    
    def forward(self, X):

        outputs = self.bert(**X,  output_hidden_states=True)
        
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        
        return output_dir
    
    def mlmForward(self, X, Y = None):
        outputs = self.bert(**X,  labels = Y)
        return outputs.loss
        
    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output



class BERT_MDF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_MDF, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1) #0.1
        self.classifier = nn.Linear(args.feat_dim, 2)  
        self.init_weights() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_hidden_states=True
        )
        # Complains if input_embeds is kept

        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClassificationHead(nn.Module):
    def __init__(self, config):
        super(BertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels-1)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BertContrastiveHead(nn.Module):
    def __init__(self, config):
        super(BertContrastiveHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BERT_KNNCL(nn.Module):

    def __init__(self, args):
        super(BERT_KNNCL, self).__init__()

        self.number_labels = args.anum_labels

        config = AutoConfig.from_pretrained(
        args.bert_model ,
        num_labels=self.number_labels,
        )
        
        self.encoder_q = BertModel.from_pretrained(args.bert_model, config=config)
        self.encoder_k = BertModel.from_pretrained(args.bert_model, config=config)

        self.classifier_liner = BertClassificationHead(config)

        self.contrastive_liner_q = BertContrastiveHead(config)
        self.contrastive_liner_k = BertContrastiveHead(config)

        self.m = 0.999
        self.T = args.temperature
        self.init_weights()  # Exec
        self.contrastive_rate_in_training = args.contrastive_rate_in_training

        # create the label_queue and feature_queue
        self.K = args.queue_size  # 7500

        self.register_buffer("label_queue", torch.randint(0, self.number_labels, [self.K]))  # Tensor:(7500,)
        self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))  # Tensor:(7500, 768)
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # Tensor(1,)
        self.top_k = args.top_k  # 25
        self.update_num = args.positive_num  # 3

        # optional and delete can improve the performance indicated
        # by some experiment
        # params_to_train = ["layer." + str(i) for i in range(0, 12)]
        # for name, param in self.encoder_q.named_parameters():
        #     param.requires_grad_(False)
        #     for term in params_to_train:
        #         if term in name:
        #             param.requires_grad_(True)

    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr: ptr + batch_size, :] = keys
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_pos_neg_sample(self, liner_q, label_q):
        label_queue = self.label_queue.clone().detach()  # K
        feature_queue = self.feature_queue.clone().detach()  # K * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])  # batch_size * K * hidden_size

        # 2.caluate sim
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3. get index of postive and neigative 
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4.another option 
        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]  # self.topk = 25
        pos_sample = pos_sample_top_k
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.top_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def l2norm(self, x: torch.Tensor):
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x

    def forward_no_multi_v2(self,
                            query,
                            positive_sample=None,
                            negative_sample=None,
                            ):
        labels = query["labels"]
        labels = labels.view(-1)

        with torch.no_grad():
            self.update_encoder_k()
            update_sample = self.reshape_dict(positive_sample)
            bert_output_p = self.encoder_k(**update_sample)
            update_keys = bert_output_p[1]
            update_keys = self.contrastive_liner_k(update_keys)
            update_keys = self.l2norm(update_keys)
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.repeat([1, self.update_num])
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)

        query.pop('labels')

        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        liner_q = self.l2norm(liner_q)
        logits_cls = self.classifier_liner(q)

        if self.number_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.number_labels - 1), labels)

        logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * self.contrastive_rate_in_training + \
                   loss_cls * (1 - self.contrastive_rate_in_training)
        else:
            loss = loss_cls

        return SequenceClassifierOutput(
            loss=loss,
        )

    def forward(self,
                query,  # batch_size * max_length
                mode,
                positive_sample=None,  # batch_size * max_length
                negative_sample=None,  # batch_size * sample_num * max_length
                ):
        if mode == 'train':
            return self.forward_no_multi_v2(query=query, positive_sample=positive_sample,
                                            negative_sample=negative_sample)
        elif mode == 'validation':
            labels = query['labels']
            query.pop('labels')
            seq_embed = self.encoder_q(**query)[1]

            logits_cls = self.classifier_liner(seq_embed)
            probs = torch.softmax(logits_cls, dim=1)
            return torch.argmax(probs, dim=1).tolist(), labels.cpu().numpy().tolist()
        elif mode == 'test':

            query.pop('labels')
            seq_embed = self.encoder_q(**query)[1]
            logits_cls = self.classifier_liner(seq_embed)

            probs = torch.softmax(logits_cls, dim=1)
            return probs, seq_embed
        else:
            raise ValueError("undefined mode")

