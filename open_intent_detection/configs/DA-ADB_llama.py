class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            llama_model (directory): The path for the pre-trained llama model.
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            scale (float): The scale factor of the cosine classifier.
            lr_boundary (float): The learning rate of the decision boundary.
            lr (float): The learning rate of backbone.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'llama_model': "/home/sharing/disk1/pretrained_embedding/llama/llama",
            'num_train_epochs':100,
            'num_labels': None,
            'max_seq_length': None, 
            'freeze_backbone_parameters': False,
            'feat_dim': 4096,
            'warmup_proportion': 0.1,
            'scale': 4,
            'lr_boundary': 0.05,
            'lr': 5e-8, 
            'activation': 'relu',
            'train_batch_size': 32,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patient': 10,

        }
        print("Hyper-parameters: ", hyper_parameters)

        return hyper_parameters
