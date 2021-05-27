class OpenMax_Param():
    
    def __init__(self):
        
        self.hyper_param = self.get_hyper_parameters()

    def get_hyper_parameters(self):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs: The training epochs.
            max_seq_len (int): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            loss_fct (str): The loss function for training.
            weibull_tail_size (int): The factor of weibull model.
            alpharank (int): The factor of alpha rank.
            distance_type (str): The distance_type.
            threshold (float): The probability threshold for detecting the open samples.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'max_seq_length': None, 
            'freeze_bert_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr': 2e-5, 
            'loss_fct': 'cross_entropy',
            'weibull_tail_size': 20,
            'alpharank': 10,
            'distance_type': 'cosine',
            'threshold': 0.5,
            'train_batch_size': 128,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 10

        }

        return hyper_parameters