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
            'glove_model': '/home/sharing/disk1/pretrained_embedding/glove',
            'max_num_words': 10000, 
        }

        return hyper_parameters