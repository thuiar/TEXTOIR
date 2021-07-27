import logging
from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix


class DECManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.sae = model.sae
        self.tfidf_train, self.tfidf_test = data.dataloader.tfidf_train, data.dataloader.tfidf_test

        self.num_labels = data.num_labels
        self.test_y = data.dataloader.test_true_labels

        if args.train:
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
    
