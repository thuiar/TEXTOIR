class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):

        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'num_train_epochs': 100,
            'num_labels': None,
            'max_seq_length': None, 
            'freeze_backbone_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr': 1e-05, 
            'activation': 'relu',
            'train_batch_size': 128,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 100,
            'warmup_proportion':0.1,

            'n_neighbors': 20,
            'contamination': 0.05,

            "temperature": 0.5 ,
            "negative_num":96,
            "positive_num":3,
            "end_k" : 1,
            "m" : 0.999,
            "contrastive_rate_in_training" : 0.1,
            "warmup_steps" : 0,
            "clip" : 0.25,
            "weight_decay" : 0.0001,
            "anum_labels" : None,
            "adam_beta1" :0.9,
            "adam_beta2" : 0.98,

            "top_k" : 25,
            "queue_size": 7500 

        }

        if args.dataset == "banking":
            if args.known_cls_ratio == 0.25:
                hyper_parameters["queue_size"]=6500
            elif args.known_cls_ratio == 0.5:
                hyper_parameters["queue_size"]=8000
                hyper_parameters["top_k"]=30

        if args.dataset == "oos":
            hyper_parameters["queue_size"]=6500
            hyper_parameters["top_k"]=15


        return hyper_parameters