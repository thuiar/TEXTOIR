import torch

from peft import (  
    LoraConfig,
    get_peft_model,
)

from torch import nn
from transformers import AutoModelForCausalLM

from .bert import CosNorm_Classifier

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class LLAMA_lora_Disaware(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_labels = args.num_labels
        self.llama = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            return_dict=True,
            load_in_8bit=False,
            device_map=args.device,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.llama.config.pad_token_id  = 0  # unk
        self.llama.config.bos_token_id = 1
        self.llama.config.eos_token_id = 2
        #self.llama.eval()
        target_modules=[ "q_proj", "v_proj"]
        config = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        print("lora", config)
        self.llama = get_peft_model(self.llama, config)
        hidden_dropout_prob = 0.1
        hidden_size = self.llama.config.hidden_size
        hidden_size_2 = hidden_size // 2
        self.dense = nn.Linear(hidden_size, hidden_size).half()
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(hidden_dropout_prob).half()
        self.dense = self.dense.to(args.device)
        self.activation = self.activation.to(args.device)
        self.dropout = self.dropout.to(args.device)
        #self.init_weights()
        self.cosnorm_classifier = CosNorm_Classifier(
            hidden_size, args.num_labels, args.scale, args.device)


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, dist_infos = None):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True )
        encoded_layer_ = outputs.hidden_states[-1].mean(dim=1)
        
        #input_data = input_data.float()
        pooled_output = self.dense(encoded_layer_)
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