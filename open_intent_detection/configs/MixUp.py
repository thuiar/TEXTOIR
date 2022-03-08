class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs: The training epochs.
            max_seq_len (int): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            threshold (float): The probability threshold for detecting the open samples.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            wait_patient (int): Patient steps for Early Stop.
            lr_mlp (float): The learning rate of mlp.
            multiple_convex (int): The multiple of gen ood of training stage.
            multiple_convex_eval (int): The multiple of gen ood of eval stage.
        """
        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'num_train_epochs': 100,
            'max_seq_length': None, 
            'freeze_backbone_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr': 2e-5, 
            'activation': 'relu',
            'train_batch_size': 64,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 10,
            'multiple_convex': 1,
            'multiple_convex_eval': 1,
            'dataset_neg': 'SQUAD',
            'temp': 0.2

        }

        return hyper_parameters
