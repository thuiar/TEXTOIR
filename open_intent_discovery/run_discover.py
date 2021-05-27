from open_intent_discovery.init_parameters import Param
from open_intent_discovery.dataloader import *
from open_intent_discovery.utils import *


def run(args):
    test = False
    test = True
    if test:
        print("Notice: ****This is test mode****")
        args.dataset = 'banking'
        args.labeled_ratio = 0.8
        args.labeled_ratio = 1.0
        args.known_cls_ratio = 0.5
        args.num_train_epochs = 1
        args.method = "SAE"
        args.setting = 'unsupervised'
        # args.train_discover = True
        # args.save_discover = True
        args.gpu_id = '1'
    print("Setting-----"+args.setting+'\n')
    print("Method_test-----"+args.method+"\n")

    check_inputs(args)
    
    print('Data Preparation...')
    if args.setting == 'unsupervised':
        data = Unsup_Data(args)

    elif args.setting == 'semi_supervised':
        data = Data(args)
    
    manager = get_manager(args, data)
    
    if args.train_discover:
        print('Training Begin...')
        manager.train(args, data)
        print('Training Finished...')

    print('Evaluation begin...')
    outputs = manager.evaluation(args, data)
    print('Evaluation finished...')
    open_ids = [idx for idx, label in enumerate(outputs[1]) if data.all_label_list[label] not in data.known_label_list] 
    open_preds = outputs[0][open_ids]
    open_trues = outputs[1][open_ids]
    results = clustering_score(open_trues, open_preds)
    print('run_discover.py: 44: ', results)
    # discover_centers(args, data, outputs)
    # print('Keywords Extraction begin...')
    # keywords_extraction(args, data, outputs)
    

    save_discover_backend_results(manager, args, data)
    save_discover_frontend_results(args, data, outputs)
    
    debug(data, manager, args)
    print('Open Intent Discovery Finished...')


if __name__ == '__main__':
    print('Open Intent Discovery Begin...')

    print('Parameters Initialization...')
    param = Param()
    args = param.args

    run(args)