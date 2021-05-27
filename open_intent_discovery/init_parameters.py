import argparse
import importlib
import sys
import os
rootPath = '/home/zk/Baselines/TEXTOIR'
class Param:

    def __init__(self, args_pipe = None):
        
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()  
        args_dict = all_args.__dict__

        if args_pipe is not None:
            args_pipe_dict = args_pipe.__dict__
            for key in args_pipe_dict:
                if key in args_dict.keys():
                    args_dict[key] = args_pipe_dict[key]
                else:
                    args_dict[key] = args_pipe_dict[key]
        
        mtd = all_args.method == all_args.discover_method
        if mtd:
            method = eval(all_args.method)
            print("only discover")
        else:
            method = eval(all_args.discover_method)
            print("pipline discover")
        self.args = method(all_args).args

    def all_param(self, parser):

        ##################################common parameters####################################
        parser.add_argument("--dataset", default=None, type=str, help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--method", type=str, default='DeepAligned', help="which method to use")

        parser.add_argument("--backbone", default='bert', type=str, help="which model to use")

        parser.add_argument("--cluster_num_factor", default=1, type=int, help="The factor (magnification) of the number of clusters K.")

        parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        
        parser.add_argument('--type', type=str, default='open_intent_discovery', help="Type for methods")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
        
        parser.add_argument('--setting', type=str, default='semi_supervised', help="Type for clustering methods.")

        parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")

        parser.add_argument("--data_dir", default=sys.path[0]+'/data', type=str,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
         
        parser.add_argument("--gpu_id", type=str, default='1', help="Select the GPU id")

        parser.add_argument("--frontend_dir", type=str, default=os.path.join(sys.path[0],'../frontend/static/jsons') , help="the path of the frontend")

        parser.add_argument("--train_discover", action="store_true", help="Whether train the model")

        #####################################unsupervised parameters##################################
        parser.add_argument("--max_num_words", default=10000, type=int, help="The maximum number of words.")
        
        # parser.add_argument("--feat_dim", default=2000, type=int, help="The feature dimension.")

        parser.add_argument("--glove_model", default='/home/sharing/disk2/zhl_backup/pretrained_models/glove', type=str, help="The path for the pre-trained bert model.")

        #####################################DEC & DCN parameters###########################################
        parser.add_argument("--maxiter", default=12000, type=int, help="The training epochs for DEC.")

        parser.add_argument("--update_interval", default=100, type=int, help="The training epochs for DEC.")

        parser.add_argument("--batch_size", default=256, type=int, help="The training epochs for DEC.")

        parser.add_argument("--tol", default=0.001, type=float, help="The tolerance threshold to stop training for DEC.")

        #####################################DTC-BERT parameters###########################################
        parser.add_argument("--rampup_coefficient", default=10.0, type=float, help="The rampup coefficient.")

        parser.add_argument("--rampup_length", default=5, type=int, help="The rampup length.")

        parser.add_argument("--num_warmup_train_epochs", default=1, type=int, help="The number of warm-up training epochs.")

        parser.add_argument("--alpha", default=0.6, type=float)

        ##############BERT parameters#####################

        parser.add_argument("--train_data_dir", default= os.path.join(rootPath, 'models'), type=str, 
                            help="The output directory where all train data will be written.") 

        parser.add_argument("--bert_model", default="/home/sharing/disk2/zhl_backup/pretrained_models/uncased_L-12_H-768_A-12", type=str, help="The path for the pre-trained bert model.")

        parser.add_argument("--pretrain", action="store_true", default = 'pretrain', help="Pretrain the model")

        parser.add_argument("--model_dir", default='models', type=str, 
                            help="The output directory where the model predictions and checkpoints will be written.") 
        
        parser.add_argument("--max_seq_length", default=None, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")

        parser.add_argument("--warmup_proportion", default=0.1, type=float)

        parser.add_argument("--freeze_bert_parameters", action="store_true", default = "freeze", help="Freeze the last parameters of BERT")

        parser.add_argument("--save_discover", action="store_true", help="save trained-model for open intent discovery")

        parser.add_argument("--lr", default=5e-5, type=float,
                            help="The learning rate of BERT.")    

        parser.add_argument("--lr_pre", default=5e-5, type=float, help="The learning rate for pre-training.")

        parser.add_argument("--num_train_epochs", default=100.0, type=float,
                            help="Total number of training epochs to perform.") 
        
        parser.add_argument("--train_batch_size", default=128, type=int,
                            help="Batch size for training.")
        
        parser.add_argument("--eval_batch_size", default=64, type=int,
                            help="Batch size for evaluation.")    
        
        parser.add_argument("--wait_patient", default=20, type=int,
                            help="Patient steps for Early Stop.")    

        return parser

class KM:
    def __init__(self, args):
        args.feat_dim = 2000
        self.args = args

class AG:
    def __init__(self, args):
        args.feat_dim = 2000
        self.args = args

class DEC:
    def __init__(self, args):
        args.feat_dim = 2000
        args.maxiter = 500
        args.backbone = 'SAE'
        self.args = args

class DCN:
    def __init__(self, args):
        args.feat_dim = 2000
        args.maxiter = 12000
        self.args = args

class SAE:
    def __init__(self, args):
        args.feat_dim = 2000
        args.num_train_epochs = 150
        args.batch_size = 4096
        args.backbone = 'SAE'
        self.args = args

class KCL_BERT:
    def __init__(self, args):
        self.args = args

class DTC_BERT:
    def __init__(self, args):
        args.lr_pre = 2e-5
        self.args = args

class MCL_BERT:
    def __init__(self, args):
        self.args = args

class DeepAligned:
    def __init__(self, args):
        self.args = args

class CDACPlus:
    def __init__(self, args):
        args.num_train_epochs = 46
        args.train_batch_size = 256
        args.eval_batch_size = 256
        args.lr = 5e-5
        args.u = 0.95
        args.l = 0.455
        self.args = args

