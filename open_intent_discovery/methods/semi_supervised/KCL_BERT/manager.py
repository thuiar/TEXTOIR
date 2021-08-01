import logging
from tqdm import trange, tqdm

def Class2Simi(x,mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

class KCLManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.model = model.model 
        self.optimizer = model.optimizer
        self.device = model.device

        self.train_dataloader = data.dataloader.train_loader

        if args.train:

            from backbones.bert import BertForKCL_Similarity
            args.backbone = 'bert_KCL_simi'
            self.pretrain_model = model.set_model(args, data, 'bert')
            self.pretrain_optimizer = model.set_optimizer(self.pretrain_model, )
            self.best_eval_score = 0
            self.simi_model = self.train_simi(args, data)
            self.model = KCL.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)

        else:
            self.simi_model = SimiModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.simi_model = restore_model(self.simi_model, self.pretrain_model_dir)
            self.model = KCL.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)
            self.model = restore_model(self.model, self.model_dir)
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.simi_model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)


    def pre_train(self, args, data):
    
        manager_p = SimiModelManager(args, data)
        manager_p.train(args, data)
        print('Pretraining finished...')

        if args.save_discover:
            save_model(manager_p.model, self.pretrain_model_dir)
        
        return manager_p.model

    def get_preds_labels(self, args, dataloader, model):
        
        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                simi = self.prepare_task_target(batch, self.simi_model)
                preds, feature = model(input_ids, segment_ids, input_mask)
    
            total_labels = torch.cat((total_labels, label_ids))
            total_preds = torch.cat((total_preds, preds))
            total_features = torch.cat((total_features, feature))

        y_true = total_labels.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        feats = total_features.cpu().numpy()

        return y_true, y_pred, feats
    
    def prepare_task_target(self, batch, model):

        model.eval()
        input_ids,input_mask,segment_ids,label_ids = batch
        target = model(input_ids, segment_ids, input_mask)
        target = target.float()
        target[target == 0] = -1

        return target.detach()

    def train(self, args, data): 

        best_model = None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(data.train_dataloader, desc="Training(All)"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                with torch.no_grad():
                    simi = self.prepare_task_target(batch, self.simi_model)
                
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', simi = simi)
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss',tr_loss)

            y_true, y_pred, feats = self.get_preds_labels(args, data.eval_dataloader, self.model)
            results = clustering_score(y_true, y_pred)
            eval_score = results['NMI']
            print('eval_score',eval_score)

            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
            
            #########debug
            # y_true, y_pred, feats = self.get_preds_labels(data.test_dataloader, self.model)
            # results = clustering_score(y_true, y_pred)
            # print('test_score', results)

        self.model = best_model

        if args.save_discover:
            save_model(self.model, self.model_dir)

    def test(self, args, data, show=True):
    
        y_true, y_pred, feats = self.get_preds_labels(args, data.test_dataloader, self.model)
        results = clustering_score(y_true, y_pred) 
        
        if show:
            print('results',results)

        self.test_results = results

        return y_pred, y_true, feats
