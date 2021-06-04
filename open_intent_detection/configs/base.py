import argparse
import sys
import os
from .ADB import ADB_Param
from .MSP import MSP_Param
from .DeepUnk import DeepUnk_Param
from .DOC import DOC_Param
from .OpenMax import OpenMax_Param
from utils.functions import Storage

param_map = {
    'ADB': ADB_Param, 'MSP': MSP_Param, 'DeepUnk': DeepUnk_Param, 'DOC': DOC_Param, 'OpenMax': OpenMax_Param
}


class ParamManager:
    
    def __init__(self):
        
        args = self.add_base_param()
        
        output_path_param = self.add_output_path_param(args)

        method_param = self.get_method_param(args)

        self.args = Storage(
                                dict(
                                        vars(args),
                                        **output_path_param,
                                        **method_param
                                     )
                            )

    def add_base_param(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('--type', type=str, default='open_intent_detection', help="Type for methods")

        parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--method", type=str, default='ADB', help="which method to use")

        parser.add_argument("--train", action="store_true", help="Whether train the model")

        parser.add_argument("--save_model", action="store_true", help="save trained-model for open intent detection")

        parser.add_argument("--backbone", type=str, default='roberta', help="which model to use")

        parser.add_argument("--num_train_epochs", type=int, default=100, help = "The number of training epochs.")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument("--gpu_id", type=str, default='1', help="Select the GPU id")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
        
        parser.add_argument("--data_dir", default=sys.path[0]+'/../data', type=str,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
 
        parser.add_argument("--output_dir", default= '/home/sharing/disk2/zhouqianrui/TEXTOIR/outputs', type=str, 
                            help="The output directory where all train data will be written.") 

        parser.add_argument("--model_dir", default='models', type=str, 
                            help="The output directory where the model predictions and checkpoints will be written.") 

        parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")
        
        args = parser.parse_args()

        return args

    def get_method_param(self, args):

        method_param = param_map[args.method]
        method_args = method_param()

        return method_args.hyper_param

    def add_output_path_param(self, args):
        
        task_output_dir = os.path.join(args.output_dir, args.type)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
        method_output_name = "_".join([str(x) for x in concat_names])

        method_output_dir = os.path.join(task_output_dir, method_output_name)
        if not os.path.exists(method_output_dir):
            os.makedirs(method_output_dir)

        model_output_dir = os.path.join(method_output_dir, args.model_dir)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        output_path_param = {
            'method_output_dir': method_output_dir,
            'model_output_dir': model_output_dir,
        }

        return output_path_param