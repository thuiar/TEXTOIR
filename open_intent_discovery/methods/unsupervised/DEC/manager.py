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
        self.y_test = None
        self.test_results = None
        self.num_labels = data.num_labels

        #Save models and trainined data
        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.backbone]
        output_file_name = "_".join([str(x) for x in concat_names])
        self.output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
        self.output_file_dir = os.path.join(self.output_dir, args.save_results_path)
        self.model_dir = os.path.join(self.output_dir, args.model_dir)

        if args.train_discover:
            self.model = None
        else:
            self.sae_emb.load_weights(self.model_dir + '_SAE.h5')
            clustering_layer = Backbone.ClusteringLayer(self.num_labels, name='clustering')(self.sae_emb.layers[3].output)
            self.model = Model(inputs=self.sae_emb.input, outputs=clustering_layer)
            self.model.load_weights(self.model_dir + '_DEC.h5')

    def train(self, args, data):

        print("Training: SAE(emb)")
        
        self.sae_emb.fit(self.tfidf_train, self.tfidf_train, epochs = args.num_train_epochs, batch_size = args.batch_size, shuffle=True, 
                    validation_data=(self.tfidf_test, self.tfidf_test), verbose=1)

        emb_train, emb_test = data.get_sae(args, self.sae_emb, self.tfidf_train, self.tfidf_test)
        
        clustering_layer = Backbone.ClusteringLayer(self.num_labels, name='clustering')(self.sae_emb.layers[3].output)
        model = Model(inputs=self.sae_emb.input, outputs=clustering_layer)
        model.compile(optimizer=SGD(0.001, 0.9), loss='kld')
        km = KMeans(n_clusters=self.num_labels, n_init=20, n_jobs=-1)
        
        y_pred = km.fit_predict(emb_train)
        y_pred_last = np.copy(y_pred)
        model.get_layer(name='clustering').set_weights([km.cluster_centers_])

        index = 0
        loss = 0
        index_array = np.arange(self.tfidf_train.shape[0])
        
        for ite in range(int(args.maxiter)):

            if ite % args.update_interval == 0:
                q = model.predict(self.tfidf_train, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if data.y_train is not None:
                    results = clustering_score(data.y_train, y_pred)
                    print('Iter=', ite, results, 'loss=', np.round(loss, 5))
                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < args.tol:
                    print('delta_label ', delta_label, '< tol ', args.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * args.batch_size: min((index+1) * args.batch_size, self.tfidf_train.shape[0])]
            loss = model.train_on_batch(x=self.tfidf_train[idx], y=p[idx])
            index = index + 1 if (index + 1) * args.batch_size <= self.tfidf_train.shape[0] else 0

        self.model = model
        
        if args.save_discover:

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            sae_model_dir = self.model_dir + '_SAE.h5'
            dec_model_dir = self.model_dir + '_DEC.h5'

            self.sae_emb.save_weights(sae_model_dir)
            self.model.save_weights(dec_model_dir)


    def evaluation(self, args, data, show=False):
        feats = self.sae_emb(self.tfidf_test)
        feats = feats.numpy()
        q = self.model.predict(self.tfidf_test, verbose = 0)
        y_pred = q.argmax(1)
        y_true = data.y_test
        results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true,y_pred) 

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])
        
        if show:
            print('results',results)
            print('confusion matrix', cm)

        self.test_results = results
        return y_pred, y_true, feats

    def save_results(self, args, data):
        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        #save known intents
        np.save(os.path.join(self.output_file_dir, 'labels.npy'), data.label_list)

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
        
    
