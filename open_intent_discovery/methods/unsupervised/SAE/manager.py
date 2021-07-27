import logging
import os
from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix


class SAEManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.sae = model.sae
        self.tfidf_train, self.tfidf_test = data.dataloader.tfidf_train, data.dataloader.tfidf_test

        self.num_labels = data.num_labels
        self.test_y = data.dataloader.test_true_labels

    def train(self, args, data):
        
        self.logger.info('SAE (emb) training start...')
        
        self.sae.fit(self.tfidf_train, self.tfidf_train, epochs = args.num_train_epochs, batch_size = args.batch_size, shuffle=True, 
                    validation_data=(self.tfidf_test, self.tfidf_test), verbose=1)

        if args.save_model:
            save_path = os.path.join(args.model_output_dir, args.model_name)
            self.sae.save_weights(save_path)

    
    def test(self, args, data, show=False):
        
        from backbones.sae import get_sae
        
        if not args.train:
            save_path = os.path.join(args.model_output_dir, args.model_name)
            self.sae.load_weights(args.model_output_dir)

        sae_emb_train, sae_emb_test = get_sae(args, self.sae, self.tfidf_train, self.tfidf_test)
        
        self.logger.info('K-Means start...')
        from sklearn.cluster import KMeans


        km = KMeans(n_clusters= self.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(sae_emb_train)
        self.logger.info('K-Means finished...')

        y_pred = km.predict(sae_emb_test)
        y_true = self.test_y
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
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
        
    
