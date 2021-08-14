import argparse
import sys
import os
import importlib
import json
from easydict import EasyDict

class ParamManager:
    
    def __init__(self, args):

        output_path_param = self.add_output_path_param(args)
        
        if args.save_frontend_results:
            self.frontend_param = self.add_frontend_path_param(args)

            method_param = self.get_method_param(args)

            self.args = EasyDict(
                                    dict(
                                            vars(args),
                                            **self.frontend_param,
                                            **output_path_param,
                                            **method_param
                                        )
                                )

        else:

            method_param = self.get_method_param(args)

            self.args = EasyDict(
                                    dict(
                                            vars(args),
                                            **output_path_param,
                                            **method_param
                                        )
                                )

    def get_method_param(self, args):
        
        if args.config_file_name.endswith('.py'):
            module_name = '.' + args.config_file_name[:-3]
        else:
            module_name = '.' + args.config_file_name

        config = importlib.import_module(module_name, 'configs')

        method_param = config.Param
        method_args = method_param(args)

        if args.save_frontend_results:
            
            if os.path.exists(self.frontend_param["config_results_dir"]):

                with open(self.frontend_param["config_results_dir"]) as f:
                    config_dicts = json.load(f)

                flag = True
                for key in config_dicts:
                    if key not in method_args.hyper_param:
                        flag = False
                        break
                
                if flag:
                    method_args.hyper_param = config_dicts

        return method_args.hyper_param

    def add_frontend_path_param(self, args):

        save_dir = os.path.join(args.frontend_result_dir, args.type) 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        analysis_file_name = args.method + '_analysis.json'
        test_file_name = 'json_test_results.json'
        detection_file_name = 'json_detection_results.json'
        config_file_name = 'config.json'

        paths = []
        for save_file_name in [analysis_file_name, test_file_name, detection_file_name, config_file_name]:
            results_path = os.path.join(save_dir, save_file_name)
            if not os.path.exists(results_path):
                f = open(results_path, 'w')

            paths.append(results_path)

        frontend_path_param = {
            'analysis_output_dir': paths[0],
            'test_results_dir': paths[1],
            'train_results_dir': paths[2],
            'config_results_dir': paths[3]
        }

        return frontend_path_param

    def add_output_path_param(self, args):
        
        task_output_dir = os.path.join(args.output_dir, args.type)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone, args.seed]
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