from open_intent_discovery.utils import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):
        
        
        self.tfidf_train, self.tfidf_test = data.get_tfidf(args)
        self.sae_emb = Backbone.get_stacked_autoencoder(self.tfidf_train.shape[1])

        self.km = None
        self.emb_test = None
        self.y_test = None
        self.test_results = None
        self.num_labels = data.num_labels

        #Save models and trainined data
        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.backbone]
        output_file_name = "_".join([str(x) for x in concat_names])
        self.output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
        self.output_file_dir = os.path.join(self.output_dir, args.save_results_path)
        self.model_dir = os.path.join(self.output_dir, args.model_dir)


    def train(self, args, data):
        
        print("Training: SAE(emb)")
        
        self.sae_emb.fit(self.tfidf_train, self.tfidf_train, epochs = args.num_train_epochs, batch_size = args.batch_size, shuffle=True, 
                    validation_data=(self.tfidf_test, self.tfidf_test), verbose=1)

        if args.save_discover:
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            self.sae_emb.save_weights(self.model_dir)

    def evaluation(self, args, data, show=False):
        
        if args.train_discover:
            emb_train, emb_test = data.get_sae(args, self.sae_emb, self.tfidf_train, self.tfidf_test)
        else:
            self.sae_emb.load_weights(self.model_dir)
            emb_train, emb_test = data.get_sae(args, self.sae_emb, self.tfidf_train, self.tfidf_test)

        km = KMeans(n_clusters= self.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(emb_train)
        y_pred = km.predict(emb_test)
        y_true = data.y_test
        results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true,y_pred) 
        
        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if show:
            print('results',results)
            print('confusion matrix', cm)

        self.test_results = results
        return y_pred, y_true, emb_test

    def save_results(self, args, data):

        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        result_file = 'results.csv'
        results_path = os.path.join(args.train_data_dir, args.type, result_file)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)

        static_dir = os.path.join(args.frontend_dir, args.type)
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        #save true_false predictions
        predict_t_f, predict_t_f_fine = cal_true_false(self.true_labels, self.predictions)
        csv_to_json(results_path, static_dir)

        tf_overall_path = os.path.join(static_dir, 'ture_false_overall.json')
        tf_fine_path = os.path.join(static_dir, 'ture_false_fine.json')

        results = {}
        results_fine = {}
        key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.cluster_num_factor) + '_' + str(args.method)
        if os.path.exists(tf_overall_path):
            results = json_read(tf_overall_path)

        results[key] = predict_t_f

        if os.path.exists(tf_fine_path):
            results_fine = json_read(tf_fine_path)
        results_fine[key] = predict_t_f_fine

        json_add(results, tf_overall_path)
        json_add(results_fine, tf_fine_path)

        print('test_results', data_diagram)
        
    
