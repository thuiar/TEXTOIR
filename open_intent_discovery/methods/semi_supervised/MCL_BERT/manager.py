import torch
from tqdm import trange, tqdm
from utils.metrics import clustering_score

class MCLManager:
    
    def __init__(self, args, data):
        
        self.num_labels = data.num_labels   

        self.model = MCL.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)

        if args.train:
            self.best_eval_score = 0
        else:
            self.restore_model(args)

    def get_outputs(self, args, dataloader, model):
        
        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                preds, feature = model(input_ids, segment_ids, input_mask, ext_feats = True)
    
            total_labels = torch.cat((total_labels, label_ids))
            total_preds = torch.cat((total_preds, preds))
            total_features = torch.cat((total_features, feature))

        y_true = total_labels.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        feats = total_features.cpu().numpy()

        return y_true, y_pred, feats

    def test(self, args, data, show=True):

        y_true, y_pred, feats = self.get_preds_labels(args, data.test_dataloader, self.model)
        results = clustering_score(y_true, y_pred)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]:i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])
        
        self.predictions = list([data.all_label_list[idx] for idx in y_pred])
        self.true_labels = list([data.all_label_list[idx] for idx in y_true])
        
        cm = confusion_matrix(y_true,y_pred) 
        
        if show:
            print('results',results)
            print('y_pred',y_pred)
            print('y_true',y_true)
            print('confusion matrix',cm)

        self.test_results = results
        return y_pred, y_true, feats

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
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss',tr_loss)

            y_true, y_pred, _ = self.get_preds_labels(args, data.eval_dataloader, self.model)
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

        self.model = best_model
        
        if args.save_discover:
            self.save_model(args)


