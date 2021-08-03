import os
class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            SAE_feats_path (directory): The path for pre-trained stacked auto-encoder features.
            num_train_epochs_SAE (int): The number of epochs for training stacked auto-encoder.
            num_train_epochs_DCN (int): The number of epochs for training DCN model.
            update_interval (int): The number of intervals between contiguous updates.
            lr (float): The learning rate for training DCN.
            momentum (float): The momentum value of SGD optimizer.
            tol (float): The tolerance threshold to stop training for DCN.
            model_name (str): The name of the DCN model (saved in the format of keras).
        """
        hyper_parameters = {
            'SAE_feats_path': os.path.join('_'.join([str(x) for x in ['SAE', args.dataset, 'sae', str(args.seed)]]), 'models', 'SAE.h5'),
            'num_train_epochs_SAE': 5000,
            'num_train_epochs_DCN': 12000,
            'update_interval': 100,
            'batch_size': 256,
            'lr': 0.001,
            'momentum': 0.9,
            'tol': 0.01,
            'model_name': 'DCN.h5'
        }

        return hyper_parameters