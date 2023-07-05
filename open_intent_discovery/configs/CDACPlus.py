class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs (int): The number of training epochs.
            num_refine_epochs (int): The number of refining epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_bert_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            u (float): The upper bound of the dynamic threshold.
            l (float): The lower bound of the dynamic threshold.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'pretrained_bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'num_labels': None,
            'num_train_epochs': 49,
            'num_refine_epochs': 100,
            'max_seq_length': None, 
            'freeze_bert_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr': 5e-5, 
            'activation': 'tanh',
            'u': 0.95,
            'l': 0.455,
            'train_batch_size': 256,
            'eval_batch_size': 256,
            'test_batch_size': 64,
            'wait_patient': 5
        }

        return hyper_parameters