import argparse
import sys
import os
from .ADB import ADB_Param
from .MSP import MSP_Param
from .DeepUnk import DeepUnk_Param
from .DOC import DOC_Param
from .OpenMax import OpenMax_Param
from easydict import EasyDict

param_map = {
    'ADB': ADB_Param, 'MSP': MSP_Param, 'DeepUnk': DeepUnk_Param, 'DOC': DOC_Param, 'OpenMax': OpenMax_Param
}


class ParamManager:
    
    def __init__(self, args):
        
        output_path_param = self.add_output_path_param(args)

        method_param = self.get_method_param(args)

        self.args = EasyDict(
                                dict(
                                        vars(args),
                                        **output_path_param,
                                        **method_param
                                     )
                            )

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