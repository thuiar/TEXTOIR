from operator import mod
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel,  AutoModelForMaskedLM, BertForMaskedLM
from torch.nn.parameter import Parameter
from .utils import PairEnum
from sentence_transformers import SentenceTransformer
from losses import SupConLoss

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class Bert_SCCL(BertPreTrainedModel):
    def __init__(self, config, args):
        super(Bert_SCCL, self).__init__(config)  
        self.bert = None
        self.contrast_head = None
        self.cluster_centers = None

    def init_model(self, cluster_centers=None, alpha=1.0):
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
      
    def forward(self, input_ids, attention_mask, task_type):

        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
           
            return mean_output_1, mean_output_2, mean_output_3
        
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

class BERT_MTP_Pretrain(nn.Module):
    
    def __init__(self,  args):

        super(BERT_MTP_Pretrain, self).__init__()
        self.num_labels = args.num_labels
        self.bert = AutoModelForMaskedLM.from_pretrained(args.pretrained_bert_model)
        self.dropout = nn.Dropout(0.1) #0.1
        self.classifier = nn.Linear(args.feat_dim, args.num_labels)      
    
    def forward(self, X, ):

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

class BERT_MTP(nn.Module):
    def __init__(self,  args):
        super(BERT_MTP, self).__init__()
    
        self.bert = AutoModelForMaskedLM.from_pretrained(args.pretrained_bert_model)
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(args.feat_dim, args.num_labels)      
        self.head = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(args.feat_dim, args.mlp_head_feat_dim)
        )

    def forward(self, X):
        """logits are not normalized by softmax in forward function"""
        outputs = self.bert(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features}
        output_dir["hidden_states"] = cls_embed
        
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07, device=None):
        """compute contrastive loss"""
        loss = SupConLoss()
        output = loss(embds, labels=label, mask=mask, temperature = temperature, device=device)
        return output
    
    def save_backbone(self, save_path):
        self.bert.save_pretrained(save_path)

class BERT_GCD(BertPreTrainedModel):
    
    def __init__(self,config, args):

        super(BERT_GCD, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.mlp_head = nn.Sequential(
            nn.Linear(args.feat_dim, args.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.feat_dim, args.mlp_head_feat_dim)
        )
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)
        
        return features 

class BERT_CC(BertPreTrainedModel):
    
    def __init__(self,config, args):

        super(BERT_CC, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.cluster_num = args.num_labels
        
        self.instance_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.cluster_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.cluster_num),
            nn.Softmax(dim=1)
        ) 
        
        self.init_weights()
        
    def get_features(self, h_i, h_j):
        
        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
     
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)

        return features
        
class BERTForDeepAligned(BertPreTrainedModel):
    
    def __init__(self,config, args):

        super(BERTForDeepAligned, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)      
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output

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
       
class BERT_USNID(BertPreTrainedModel):
    
    def __init__(self, config, args):

        super(BERT_USNID, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
        
        if args.pretrain or (not args.wo_self):
            self.classifier = nn.Linear(config.hidden_size, args.num_labels)
                
        self.mlp_head = nn.Linear(config.hidden_size, args.num_labels)
            
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , feature_ext = False):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)
            
        features = self.dense(features)
        pooled_output = self.activation(features)   
        pooled_output = self.dropout(features)
        
        if self.args.pretrain or (not self.args.wo_self):
            logits = self.classifier(pooled_output)
            
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            if self.args.pretrain or (not self.args.wo_self):
                return features, logits
            else:
                return features, mlp_outputs

        else:
            if self.args.pretrain or (not self.args.wo_self):
                return mlp_outputs, logits
            else:
                return mlp_outputs, mlp_outputs
            
class BERT_USNID_UNSUP(BertPreTrainedModel):
    
    def __init__(self, config, args):

        super(BERT_USNID_UNSUP, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
 
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.mlp_head = nn.Linear(config.hidden_size, args.num_labels)
            
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, weights = None,
                feature_ext = False, mode = None, loss_fct = None, aug_feats=None, use_aug = False):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)
            
        features = self.dense(features)
        pooled_output = self.activation(features)   
        pooled_output = self.dropout(features)
        
        logits = self.classifier(pooled_output)
            
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            return features, mlp_outputs
        else:
            return mlp_outputs, logits
            
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
        self.init_weights()           
        # finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(args.num_labels, args.num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext = False, u_threshold=None, l_threshold=None, mode=None,  semi=False):

        eps = 1e-10
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
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
                label_mat[label_mat!=0] = -1 
                label_mat[label_mat==0] = 1 
                label_mat[label_mat==-1] = 0 

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
                q = (q.t() / torch.sum(q, 1)).t() 
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
        self.init_weights()

        #finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(args.num_labels, args.num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct=None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = loss_fct(logits, labels)
            return loss
        else:
            q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t() 
            return logits, q

class BertForKCL_Similarity(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForKCL_Similarity,self).__init__(config)

        self.num_labels = args.num_labels
        self.bert = BertModel(config)

        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.normalization = nn.BatchNorm1d(config.hidden_size * 4)
        self.activation = activation_map[args.activation]
        
        self.classifier = nn.Linear(config.hidden_size * 4, args.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, token_type_ids = None, attention_mask=None, labels=None, loss_fct=None, mode = None):
        
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        feat1,feat2 = PairEnum(encoded_layer_12[-1].mean(dim = 1))
        feature_cat = torch.cat([feat1,feat2], 1)

        pooled_output = self.dense(feature_cat)
        pooled_output = self.normalization(pooled_output)
        pooled_output = self.activation(pooled_output)
        logits = self.classifier(pooled_output)
        
        if mode == 'train':    
            loss = loss_fct(logits.view(-1,self.num_labels), labels.view(-1))

            return loss
        else:
            return pooled_output, logits

class BertForKCL(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForKCL, self).__init__(config)

        self.num_labels = args.num_labels
        self.bert = BertModel(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, mode = None, 
        simi = None, loss_fct = None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if mode == 'train':    

            probs = F.softmax(logits,dim=1)
            prob1, prob2 = PairEnum(probs)

            loss_KCL = loss_fct(prob1, prob2, simi)
            flag = len(labels[labels != -1])

            if flag != 0:
                loss_ce = nn.CrossEntropyLoss()(logits[labels != -1], labels[labels != -1])
                loss = loss_ce + loss_KCL
            else:
                loss = loss_KCL

            return loss
        else:
            return pooled_output, logits

class BertForMCL(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForMCL, self).__init__(config)

        self.num_labels = args.num_labels
        self.bert = BertModel(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, mode = None, loss_fct = None):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = F.softmax(logits, dim = 1)

        if mode == 'train':
            
            flag = len(labels[labels != -1])
            prob1, prob2 = PairEnum(probs)
            simi = torch.matmul(probs, probs.transpose(0, -1)).view(-1)

            simi[simi > 0.5] = 1
            simi[simi < 0.5] = -1
            loss_MCL = loss_fct(prob1, prob2, simi)

            if flag != 0:

                loss_ce = nn.CrossEntropyLoss()(logits[labels != -1], labels[labels != -1])
                loss = loss_ce + loss_MCL

            else:
                loss = loss_MCL

            return loss
            
        else:
            return pooled_output, logits