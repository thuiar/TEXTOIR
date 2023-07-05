import os
class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs_SAE (int): The number of epochs for training stacked auto-encoder.
            num_train_epochs_DCN (int): The number of epochs for training DCN model.
            feat_dim (int): The feature dimension.
            update_interval (int): The number of intervals between contiguous updates.
            batch_size (int): The batch size for training.
            lr (float): The learning rate for training DCN.
            momentum (float): The momentum value of SGD optimizer.
            tol (float): The tolerance threshold to stop training for DCN.
            model_name (str): The name of the DCN model (saved in the format of keras).
        """
        hyper_parameters = {
            'num_train_epochs_SAE': 5000,
            'num_train_epochs_DCN': 12000,
            'feat_dim': 2000,
            'update_interval': 100,
            'DCN_batch_size': 256,
            'SAE_batch_size': 4096,
            'lr': 0.001,
            'momentum': 0.9,
            'tol': 0.01,
            'model_name': 'DCN.h5'
        }

        return hyper_parameters