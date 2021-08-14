class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            feat_dim (int): The feature dimension.
            batch_size (int): The batch size for training.
            model_name (str): The name of the Stacked auto-encoder model to be saved.
        """
        hyper_parameters = {
            'num_train_epochs': 5000,   
            'feat_dim': 2000,
            'batch_size': 4096,
            'model_name': 'SAE.h5'
        }

        return hyper_parameters