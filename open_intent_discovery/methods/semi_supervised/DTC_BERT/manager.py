import logging
import copy
import torch
import torch.nn.functional as F
from .pretrain import PretrainDTCManager
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from utils.metrics import clustering_score
from utils.functions import save_model, restore_model

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class DTCManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        pretrain_manager = PretrainDTCManager(args, data, model)  

        self.device = model.device

        self.train_dataloader = data.dataloader.train_unlabeled_loader

        from dataloaders.bert_loader import get_loader
        self.eval_dataloader = get_loader(data.dataloader.eval_examples, args, data.all_label_list, 'eval')
        self.test_dataloader = data.dataloader.test_loader 

        if args.train:
            
            num_train_examples = len(data.dataloader.train_unlabeled_examples)

            self.logger.info('Pre-raining start...')
            pretrain_manager.train(args, data)
            self.logger.info('Pre-training finished...')

            args.num_labels = data.num_labels
            self.model = model.set_model(args, data, 'bert')
            self.load_pretrained_model(pretrain_manager.model)

            self.initialize_centroids(args)
            
            self.warmup_optimizer = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_warmup_train_epochs, args.lr, args.warmup_proportion)

            self.logger.info('WarmUp Training start...')
            self.p_target = self.warmup_train(args)
            self.logger.info('WarmUp Training finished...')

            self.optimizer = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)

        else:
            self.model = restore_model(self.model, args.model_output_dir)


    def initialize_centroids(self, args):

        self.logger.info("Initialize centroids...")

        feats = self.get_outputs(args, mode = 'train', get_feats = True)
        km = KMeans(n_clusters=args.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)
        self.logger.info("Initialization finished...")
        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def warmup_train(self, args):
        
        probs = self.get_outputs(args, mode = 'train', get_probs = True)
        p_target = target_distribution(probs)

        for epoch in trange(int(args.num_warmup_train_epochs), desc="Warmup_Epoch"):

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Warmup_Training")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits, q = self.model(input_ids, segment_ids, input_mask)
                loss = F.kl_div(q.log(),torch.Tensor(p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.warmup_optimizer.step()
                self.warmup_optimizer.zero_grad()       

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            eval_results = {
                'loss': tr_loss, 
                'eval_score': round(eval_score, 2)
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
        
        return p_target
    

    def get_outputs(self, args, mode = 'eval', get_feats = False, get_probs = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.num_labels)).to(self.device)
        total_probs = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                logits, probs  = self.model(input_ids, segment_ids, input_mask)
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, logits))
                total_probs = torch.cat((total_probs, probs))


        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        elif get_probs:
            return total_probs.cpu().numpy()

        else:
            total_preds = total_probs.argmax(1)
            y_pred = total_preds.cpu().numpy()

            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def train(self, args, data): 

        ntrain = len(data.dataloader.train_unlabeled_examples)
        Z = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # intermediate values
        z_ema = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # temporal outputs
        z_epoch = torch.zeros(ntrain, args.num_labels).float().to(self.device)  # current outputs

        best_model = None
        best_eval_score = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits, q = self.model(input_ids, segment_ids, input_mask)
                z_epoch[step * args.train_batch_size: (step+1) * args.train_batch_size, :] = q
                kl_loss = F.kl_div(q.log(), torch.tensor(self.p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))
                
                kl_loss.backward() 
                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad() 
            
            z_epoch = torch.tensor(self.get_outputs(args, mode = 'train', get_probs = True)).to(self.device)
            Z = args.alpha * Z + (1. - args.alpha) * z_epoch
            z_ema = Z * (1. / (1. - args.alpha ** (epoch + 1)))

            if epoch % args.update_interval == 0:
                self.logger.info('updating target ...')
                self.p_target = target_distribution(z_ema).float().to(self.device) 
                self.logger.info('updating finished ...')

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            train_loss = tr_loss / nb_tr_steps

            eval_results = {
                'train_loss': train_loss, 
                'best_eval_score': best_eval_score,
                'eval_score': round(eval_score, 2),
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:
                wait += 1
                if wait >= args.wait_patient:
                    break 
        
        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):
    
        y_true, y_pred = self.get_outputs(args,mode = 'test')
        test_results = clustering_score(y_true, y_pred) 
        cm = confusion_matrix(y_true,y_pred) 
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def load_pretrained_model(self, pretrained_model):
    
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['cluster_layer', 'classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)