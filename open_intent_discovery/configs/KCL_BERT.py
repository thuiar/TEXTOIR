class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            num_train_epochs (int): The number of training epochs.
            num_pretrain_epochs (int): The number of pre-training epochs.
            num_labels (autofill): The output dimension.
            freeze_bert_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr_pre (float): The learning rate for pre-training the backbone.
            lr (float): The learning rate of backbone.
            pretrain_loss_fct (str): The loss function for pre-training.
            loss_fct (str): The loss function for training.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'pretrained_bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'max_seq_length': None, 
            'num_train_epochs': 100,
            'num_pretrain_epochs': 100,
            'num_labels': None,
            'freeze_bert_parameters': True,
            'pretrain': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr_pre': 5e-5,
            'lr': 5e-5, 
            'pretrain_loss_fct': 'CrossEntropyLoss',
            'loss_fct': 'KCL',
            'activation': 'relu',
            'train_batch_size': 128,
            'pretrain_batch_size': 128,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 10

        }

        return hyper_parameters