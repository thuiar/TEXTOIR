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
            'num_train_epochs': 5000,
            'feat_dim': 2000,
            'batch_size': 4096,
            'model_name': 'SAE.h5'
        }

        return hyper_parameters