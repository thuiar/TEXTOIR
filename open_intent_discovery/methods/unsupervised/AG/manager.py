import logging
from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix

class AGManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        self.emb_train, self.emb_test = model.emb_train, model.emb_test
        self.num_labels = data.num_labels
        self.test_y = data.dataloader.test_true_labels

    def train(self, *args):
        self.logger.info('Agglomerative Clustering does not need training...')
        pass
    
    def test(self, args, data, show = True):

        self.logger.info('Agglomerative Clustering start...')
        from sklearn.cluster import AgglomerativeClustering
        ag = AgglomerativeClustering(n_clusters = self.num_labels)
        ag.fit(self.emb_train)

        y_pred = ag.fit_predict(self.emb_test)
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
    
