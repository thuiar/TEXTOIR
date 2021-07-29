import os
class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            glove_model (directory): The path for the pre-trained glove embedding.
            max_num_words (int): The maximum number of words.
        """
        hyper_parameters = {
            'SAE_feats_path': os.path.join('_'.join([str(x) for x in ['SAE', args.dataset, 'sae', str(args.seed)]]), 'models', 'SAE.h5'),
            'num_train_epochs_DEC': 12000,
            'num_train_epochs_SAE': 5000,
            'update_interval': 100,
            'feat_dim': 2000,
            'batch_size': 256,
            'lr': 0.001,
            'momentum': 0.9,
            'tol': 0.001,
            'model_name': 'DEC.h5'
        }

        return hyper_parameters