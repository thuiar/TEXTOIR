import torch
from torch import nn
from transformers import PreTrainedModel, XLNetConfig, XLNetModel, RobertaConfig, RobertaModel, BertPreTrainedModel, BertModel

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
    
class L2_normalization(nn.Module):
    def forward(self, input):
        return l2_norm(input)

def freeze_parameters(model, backbone):
    if backbone == "bert":
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    elif backbone == "roberta":
        for name, param in model.roberta.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    elif backbone == "xlnet":
        for name, param in model.xlnet.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    return model
    
class BERT(BertPreTrainedModel):
    
    def __init__(self,config):

        super(BERT, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states = True)
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

class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    # base_model_prefix = "roberta"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Roberta(RobertaPreTrainedModel):
    
    def __init__(self,config):

        super(Roberta, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.roberta(input_ids, attention_mask, token_type_ids, output_hidden_states = True)
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


class XLNetPreTrainedModel(PreTrainedModel):
    config_class = XLNetConfig
    # base_model_prefix = "xlnet"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XLNet(XLNetPreTrainedModel):
    
    def __init__(self,config):

        super(XLNet, self).__init__(config)
        self.num_labels = config.num_labels
        self.xlnet = XLNetModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        outputs = self.xlnet(input_ids, attention_mask, token_type_ids, output_hidden_states = True)
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
