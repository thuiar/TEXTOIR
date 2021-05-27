from configs.base import ParamManager
from dataloaders.base import DataManager
from backbones.base import ModelManager
from methods import method_map
from utils.functions import save_results



def run(args, data, model):

    method_manager = method_map[args.method]
    method = method_manager(args, data, model)
    
    if args.train:
        
        print('Training Begin...')
        method.train(args, data)
        print('Training Finished...')

    print('Testing begin...')
    outputs = method.test(args, data)
    print('Testing finished...')

    save_results(args, outputs)


if __name__ == '__main__':
    
    print('Open Intent Classification Begin...')
    print('Parameters Initialization...')
    param = ParamManager()
    args = param.args

    

    print('Data and Model Preparation...')     
    data = DataManager(args)
    model = ModelManager(args, data)

    run(args, data, model)
    print('Open Intent Classification Finished...')
    

