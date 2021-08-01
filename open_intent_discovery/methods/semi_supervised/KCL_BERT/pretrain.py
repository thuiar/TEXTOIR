import logging



class PretrainKCLManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)

        self.model = SimiModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
                    
        self.num_train_optimization_steps = int(len(data.train_labeled_examples) / args.train_batch_size + 1) * args.num_train_epochs
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0

    def evaluation(self, args, data):
        self.model.eval()

        total_labels = np.array([],dtype=np.int32)
        total_preds = np.array([],dtype=np.int32)
        
        eval_examples = 0

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            target = Class2Simi(label_ids, mode='cls').detach()

            with torch.set_grad_enabled(False):
                preds = self.model(input_ids, segment_ids, input_mask, mode = 'eval')
                total_labels = np.concatenate((total_labels, target.to('cpu').numpy()))
                total_preds = np.concatenate((total_preds, preds.to('cpu').numpy()))
        
        eval_examples += input_ids.size(0) * input_ids.size(0)
        acc = round(accuracy_score(total_labels, total_preds) * 100, 2)

        return acc


    def train(self, args, data):     
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                train_target = Class2Simi(label_ids, mode='cls').detach()
                loss_fct = nn.CrossEntropyLoss()
                loss = self.model(input_ids, segment_ids, input_mask, train_target, loss_fct = loss_fct, mode = 'train')
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1 
                
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data)
            print('eval_score', eval_score)
            
            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr_pre,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer
