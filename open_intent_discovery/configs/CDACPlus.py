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
            lr_boundary (float): The learning rate of the decision boundary.
            lr (float): The learning rate of backbone.
            loss_fct (str): The loss function for training.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'max_seq_length': None, 
            'freeze_bert_parameters': True,
            'feat_dim': 768,
            'num_labels': None,
            'warmup_proportion': 0.1,
            'lr': 5e-5, 
            'u': 0.95,
            'l': 0.455,
            'activation': 'tanh',
            'train_batch_size': 256,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'num_train_epochs': 46,
            'num_refine_epochs': 100,
            'wait_patient': 5

        }

        return hyper_parameters