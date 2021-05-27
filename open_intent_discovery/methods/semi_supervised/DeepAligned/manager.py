from open_intent_discovery.utils import *
from .pretrain import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):

        self.model_dir, self.output_file_dir, self.pretrain_model_dir = set_path(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id    

        Model = Backbone.__dict__[args.backbone]
        
        if args.train_discover:
            
            self.best_eval_score = 0
            self.centroids = None
            pretrained_model = self.pre_train(args, data) 
        
            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data, pretrained_model) 
            else:
                self.num_labels = data.num_labels  
            
            self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)
            self.model = load_pretrained_model(self.model, pretrained_model)

        else:
            
            pretrained_model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.n_known_cls)
            pretrained_model = restore_model(pretrained_model, self.pretrain_model_dir)

            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data, pretrained_model) 
            else:
                self.num_labels = data.num_labels 

            self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)
            self.model = restore_model(self.model, self.model_dir)
        

        if args.freeze_bert_parameters:
            self.model = freeze_bert_parameters(self.model)   

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args)

        self.test_results, self.predictions, self.true_labels = None, None, None

    def pre_train(self, args, data):
        
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pretraining finished...')

        if args.save_discover:
            save_model(manager_p.model, self.pretrain_model_dir)

        return manager_p.model

    def get_features_labels(self, dataloader, model, args):
        
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext = True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data, pretrained_model):
        print('Predict K begin...')

        feats, _ = self.get_features_labels(data.train_dataloader, pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('Predict K finished.. K={}, mean_cluster_size={}'.format(num_labels, drop_out))
        return num_labels
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer

    def evaluation(self, args, data, show=False):

        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        km = KMeans(n_clusters = self.num_labels).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        self.test_results =  results
        if show:
            print('results', results)

        return (y_pred, y_true, feats)

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.input_ids, data.input_mask, data.segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data): 

        best_model = None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            feats, _ = self.get_features_labels(data.train_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            
            eval_score = metrics.silhouette_score(feats, km.labels_)
            print('eval_score',eval_score)
            # self.evaluation(args, data)

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break 
            
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(train_dataloader, desc="Training(All)"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_fct = nn.CrossEntropyLoss()
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, loss_fct = loss_fct, mode='train')
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss',tr_loss)
        
        self.model = best_model

        if args.save_discover:
            save_model(self.model, self.model_dir)
            


    
