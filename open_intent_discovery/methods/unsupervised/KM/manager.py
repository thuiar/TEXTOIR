from open_intent_discovery.utils import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):
    
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
    def train(self, *args):
        pass

    def evaluation(self, args, data, show=False):

        emb_train, emb_test = data.get_glove(args, data.X_train, data.X_test)
        
        print('Clustering start...')
        km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state = args.seed)
        km.fit(emb_train)
        print('Clustering finished...')

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
    # def cal_true_false(self):
            
    #     results = {"intent_class":[], "left":[], "right":[]}
    #     trues = np.array(self.true_labels)
    #     preds = np.array(self.predictions)

    #     labels = np.unique(trues)

    #     results_fine = {}
    #     label2id = {x:i for i,x in enumerate(labels)}

    #     for label in labels:
    #         pos = np.array(np.where(trues == label)[0])
    #         num_pos = int(np.sum(preds[pos] == trues[pos]))
    #         num_neg = int(np.sum(preds[pos] != trues[pos]))

    #         results["intent_class"].append(label)
    #         results["left"].append(-num_neg)
    #         results["right"].append(num_pos)

    #         tmp_list = [0] * len(labels)
            
    #         for fine_label in labels:
    #             if fine_label != label:
                    
    #                 num = int(np.sum(preds[pos] == fine_label))
    #                 tmp_list[label2id[fine_label]] = num
                    
    #         results_fine[label] = tmp_list

    #     return results, results_fine

    # def json_read(self, path):
    
    #     with open(path, 'r')  as f:
    #         json_r = json.load(f)

    #     return json_r
    
    # def json_add(self, predict_t_f, path):
        
    #     with open(path, 'w') as f:
    #         json.dump(predict_t_f, f, indent=4)


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
        key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.labeled_ratio) + '_' + str(args.method)
        if os.path.exists(tf_overall_path):
            results = json_read(tf_overall_path)

        results[key] = predict_t_f

        if os.path.exists(tf_fine_path):
            results_fine = json_read(tf_fine_path)
        results_fine[key] = predict_t_f_fine

        json_add(results, tf_overall_path)
        json_add(results_fine, tf_fine_path)
        
        print('test_results', data_diagram)
        
    
