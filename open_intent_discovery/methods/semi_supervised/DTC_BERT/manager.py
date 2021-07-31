import logging
import torch
import torch.nn.functional as F
from .pretrain import PretrainDTCManager
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from utils.metrics import clustering_score
from utils.functions import save_model, restore_model
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

class DTCManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        self.model = model.model
        self.optimizer = model.optimizer
        self.warmup_optimizer = model.set_optimizer(self.model, len(data.dataloader.train_unlabeled_examples), args.train_batch_size, \
            args.num_warmup_train_epochs, args.lr, args.warmup_proportion)
        
        self.device = model.device

        self.train_dataloader = data.dataloader.train_unlabeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader 

        pretrain_manager = PretrainDTCManager(args, data, model)  

        if args.train:

            self.logger.info('Pre-raining start...')
            pretrain_manager.train(args, data)
            self.logger.info('Pre-training finished...')

            self.pretrained_model = pretrain_manager.model
            self.load_pretrained_model(self.pretrained_model)

            args.num_labels = data.num_labels
            self.model.center = Parameter(torch.Tensor(args.num_labels, args.num_labels)).to(self.device)
            self.init_feats, self.init_centers = self.initialize_centroids(args)
            self.model.center.data = torch.tensor(self.init_centers).float().to(self.device)

            self.logger.info('WarmUp Training start...')
            self.p_target = self.warmup_train(args, data)
            self.logger.info('WarmUp Training finished...')

        else:
            self.model = restore_model(self.model, args.model_output_dir)


    def initialize_centroids(self, args):

        self.logger.info("Initialize centroids...")

        feats = self.get_outputs(args, mode = 'train', get_feats = True)
        pca = PCA(n_components=args.num_labels)
        feats = pca.fit_transform(feats)
        km = KMeans(n_clusters=args.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)

        self.logger.info("Initialization finished...")
        logits = feat2prob(torch.from_numpy(feats).to(self.device), torch.from_numpy(km.cluster_centers_).to(self.device))
        return logits, km.cluster_centers_

    def warmup_train(self, args, data):

        
        p_target = target_distribution(self.init_feats)

        for epoch in trange(int(args.num_warmup_train_epochs), desc="Epoch"):

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Warmup_Training")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                _, logits = self.model(input_ids, segment_ids, input_mask)
                q = feat2prob(logits, self.model.center)
                loss = F.kl_div(q.log(), p_target[step * args.train_batch_size: (step+1) * args.train_batch_size])

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.warmup_optimizer.step()
                self.warmup_optimizer.zero_grad()       

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            eval_results = {
                'loss': round(tr_loss, 4), 
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

        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_probs = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask)
    
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))
                probs = feat2prob(logits, self.model.center)
                total_probs = torch.cat((total_probs, probs.to(self.device)))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        elif get_probs:
            return total_probs

        else:
            total_preds = total_probs.argmax(1)
            y_pred = total_preds.cpu().numpy()

            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def train(self, args, data): 

        self.logger.info('Training begin...')

        ntrain = len(data.dataloader.train_unlabeled_examples)
        Z = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # intermediate values
        z_ema = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # temporal outputs
        z_epoch = torch.zeros(ntrain, args.num_labels).float().to(self.device)  # current outputs

        p_target = self.p_target
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                _, logits = self.model(input_ids, segment_ids, input_mask)
                q = feat2prob(logits, self.model.center)
                z_epoch[step * args.train_batch_size: (step+1) * args.train_batch_size, :] = q
                kl_loss = F.kl_div(q.log(), p_target[step * args.train_batch_size: (step+1) * args.train_batch_size])
                kl_loss.backward() 

                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad() 
            
            z_epoch = self.get_outputs(args, mode = 'train', get_probs = True)
            Z = args.alpha * Z + (1. - args.alpha) * z_epoch
            z_ema = Z * (1. / (1. - args.alpha ** (epoch + 1)))

            if epoch % args.update_interval == 0:
                self.logger.info('updating target ...')
                p_target = target_distribution(z_ema).float().to(self.device) 
                self.logger.info('updating finished ...')

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            train_loss = tr_loss / nb_tr_steps
            eval_results = {
                'train_loss': round(train_loss, 4), 
                'eval_score': round(eval_score, 2),
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def load_pretrained_model(self, pretrained_model):
    
        pretrained_dict = pretrained_model.state_dict()
        params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in params}
        self.model.load_state_dict(pretrained_dict, strict=False)

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
