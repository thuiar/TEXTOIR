class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_bert_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr_boundary (float): The learning rate of the decision boundary.
            lr (float): The learning rate of backbone.
            loss_fct (str): The loss function for training.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'num_train_epochs': 100,
            'num_labels': None,
            'max_seq_length': None, 
            'freeze_bert_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr_boundary': 0.05,
            'lr': 2e-5, 
            'loss_fct': 'CrossEntropyLoss',
            'activation': 'relu',
            'train_batch_size': 128,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 10

        }

        return hyper_parameters