import logging
import os
import numpy as np
import copy
from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.optimizers import SGD
from tqdm import trange
from configs.base import ParamManager
from utils.functions import set_seed
from backbones.sae import get_sae, ClusteringLayer
from sklearn.cluster import KMeans

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class DECManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        # self.sae = model.set_model(args, data, 'sae')

        self.tfidf_train, self.tfidf_test = data.dataloader.tfidf_train, data.dataloader.tfidf_test
        self.num_labels = data.num_labels
        self.test_y = data.dataloader.test_true_labels

        self.init_sae(args, data, model)
        set_seed(args.seed)
        if args.train:
            self.model, self.y_pred_init = self.init_model(args)

        else:
            clustering_layer = ClusteringLayer(self.num_labels, name='clustering')(self.sae.layers[3].output)
            self.model = Model(inputs=self.sae.input, outputs = [clustering_layer, self.sae.output])
            
            save_path = os.path.join(args.model_output_dir, args.model_name)
            self.logger.info('Loading models from %s' % save_path)
            self.model.load_weights(save_path)

    def init_sae(self, args, data, model):
        self.sae = model.set_model(args, data, 'sae')           
        self.sae_feats_path = os.path.join(args.model_output_dir, 'SAE.h5') 
 
        if os.path.exists(self.sae_feats_path):
            self.logger.info('Loading SAE features from %s' % self.sae_feats_path)
            self.sae.load_weights(self.sae_feats_path)
        else:
            self.logger.info('SAE (emb) training start...')  
            self.sae.fit(self.tfidf_train, self.tfidf_train, epochs = args.num_train_epochs_SAE, batch_size = args.SAE_batch_size, shuffle=True, 
                        validation_data=(self.tfidf_test, self.tfidf_test), verbose=1)
            self.logger.info('SAE (emb) training finished...') 

            if args.save_model:

                save_path = os.path.join(args.model_output_dir, 'SAE.h5')
                self.logger.info('Save models at %s', str(save_path))
                self.sae.save_weights(save_path)

    def init_model(self, args):

        sae_emb_train, sae_emb_test = get_sae(args, self.sae, self.tfidf_train, self.tfidf_test)
        clustering_layer = ClusteringLayer(self.num_labels, name='clustering')(self.sae.layers[3].output)
        model = Model(inputs=self.sae.input, outputs = clustering_layer)
        model.compile(optimizer=SGD(args.lr, args.momentum), loss='kld')

        km = KMeans(n_clusters=self.num_labels, n_init=20, n_jobs=-1, random_state=args.seed)
        y_pred = km.fit_predict(sae_emb_train)
        y_pred_last = np.copy(y_pred)
        model.get_layer(name='clustering').set_weights([km.cluster_centers_])

        return model, y_pred_last

    def train(self, args, data):

        self.logger.info('DEC training starts...')
        index = 0
        loss = 0
        index_array = np.arange(self.tfidf_train.shape[0])
        y_pred_last = self.y_pred_init

        for epoch in trange(int(args.num_train_epochs_DEC), desc="Epoch"):

            if  epoch % args.update_interval == 0:

                q = self.model.predict(self.tfidf_train, verbose=0)
                p = target_distribution(q)  

                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                
                if epoch > 0:
                    
                    self.logger.info("***** Epoch: %s*****", str(epoch + 1))
                    self.logger.info('Training Loss: %f', np.round(loss, 5))
                    self.logger.info('Delta Label: %f', delta_label)

                    if delta_label < args.tol:
                        self.logger.info('delta_label %s < %f', delta_label, args.tol)  
                        self.logger.info('Reached tolerance threshold. Stop training.')
                        break

            idx = index_array[index * args.DEC_batch_size: min((index + 1) * args.DEC_batch_size, self.tfidf_train.shape[0])]
            loss = self.model.train_on_batch(x = self.tfidf_train[idx], y = p[idx])
            index = index + 1 if (index + 1) * args.DEC_batch_size <= self.tfidf_train.shape[0] else 0

        self.logger.info('DEC training finished...')

        if args.save_model:
            save_path = os.path.join(args.model_output_dir, args.model_name)
            self.model.save_weights(save_path)


    def test(self, args, data, show=False):

        q = self.model.predict(self.tfidf_test, verbose = 0)
        y_pred = q.argmax(1)
        y_true = self.test_y

        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true,y_pred) 
        
        if show:
            self.logger.info
            self.logger.info("***** Test: Confusion Matrix *****")
            self.logger.info("%s", str(cm))
            self.logger.info("***** Test results *****")
            
            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred  

        return test_results
    
