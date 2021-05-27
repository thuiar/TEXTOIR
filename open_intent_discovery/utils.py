# -*- coding:utf-8 -*-
import itertools
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import torch.nn.functional as F
import random
import csv
import sys
import logging
import json
import importlib
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
# from keybert import KeyBERT
from sklearn.manifold import TSNE
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def debug(data, manager, args):

    print('-----------------Data--------------------')
    data_attrs = ["data_dir","n_known_cls","num_labels","all_label_list","known_label_list"]

    for attr in data_attrs:
        attr_name = attr
        attr_value = data.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Args--------------------')
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])

    print('-----------------Manager--------------------')
    manager_attrs = ["device","test_results"]

    for attr in manager_attrs:
        attr_name = attr
        attr_value = manager.__getattribute__(attr)
        print(attr_name,':',attr_value)
    
    if manager.predictions is not None:
        print('-----------------Prediction Example--------------------')
        show_num = 10
        for i,example in enumerate(data.test_examples):
            if i >= show_num:
                break
            sentence = example.text_a
            true_label = manager.true_labels[i]
            predict_label = manager.predictions[i]
            print(i,':',sentence)
            print('Pred: {}; True: {}'.format(predict_label,true_label))

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}

def produce_json(df, method, dataset, select_type, sort_type, select_terms, metricList):
    import csv
    if sort_type == 'known_cls_ratio':
        axis_len = 3
        pos_map = {'0.25':0, '0.5':1, '0.75':2}
    elif sort_type == 'cluster_num_factor':
        axis_len = 4
        pos_map = {'1':0, '2':1, '3':2, '4':3}
        
    dic = {}
    for i, dataset in dataset.items():
        for metric in metricList:
            for j, select_term in select_terms.items():
                dic_tmp = {}
                for k, method_val in method.items():
                    _list = df[ (df["dataset"].str[:]==(dataset)) & (df[select_type] == select_term) & (df["method"].str[:]==(method_val))  ].sort_values(sort_type)
                    select_tmp = _list.drop_duplicates(subset=[sort_type],keep='first')[sort_type]
                    val=[0] * axis_len
                    
                    for l,item in select_tmp.items():
                        val[pos_map[str(item)]] = '%.2f' % ( _list[ (_list[sort_type] == item) ][metric].mean() )
                    dic_tmp[method_val]=val
                # print(dic_1)
                dic['discovery_'+str(dataset)+'_'+str(select_term)+'_'+str(metric)] = dic_tmp

    return dic

def csv_to_json(csv_file, frontend_dir):
    df = pd.read_csv(csv_file)

    dataset = df.drop_duplicates(subset=['dataset'],keep='first')['dataset']
    known_cls_ratio = df.drop_duplicates(subset=['known_cls_ratio'],keep='first')['known_cls_ratio']
    labeled_ratio = df.drop_duplicates(subset=['labeled_ratio'],keep='first')['cluster_num_factor']
    method = df.drop_duplicates(subset=['method'],keep='first')['method']   

    metricList=['ACC','ARI','NMI']

    select_types = ['known_cls_ratio', 'cluster_num_factor']
    select_terms = [known_cls_ratio, labeled_ratio]
    select_files = ['json_discovery_IOKIR.json','json_discovery_IONOC.json' ]

    for i in range(len(select_types)):
        select_type = select_types[i]
        sort_type = select_types[(i + 1) % 2]
        select_term = select_terms[i]
        select_file = select_files[i] 
        
        dic = produce_json(df, method, dataset, select_type,  sort_type, select_term, metricList)
        select_path = os.path.join(frontend_dir, select_files[(i + 1) % 2] )
        with open(select_path,'w+') as f:
            json.dump(dic,f,indent=4)

def json_read(path):
    
    with open(path, 'r')  as f:
        json_r = json.load(f)

    return json_r


def json_add(predict_t_f, path):
    
    with open(path, 'w') as f:
        json.dump(predict_t_f, f, indent=4)

def cal_true_false(true_labels, predictions):
            
    results = {"intent_class":[], "left":[], "right":[]}
    trues = np.array(true_labels)
    preds = np.array(predictions)

    labels = np.unique(trues)

    results_fine = {}
    label2id = {x:i for i,x in enumerate(labels)}

    for label in labels:
        pos = np.array(np.where(trues == label)[0])
        # num_pos = int(np.sum(preds[pos] == trues[pos]))
        # num_neg = int(np.sum(preds[pos] != trues[pos]))
        num_pos = int(np.sum(preds[pos] == label))
        num_neg = int(np.sum(preds[pos] != label))

        results["intent_class"].append(label)
        results["left"].append(-num_neg)
        results["right"].append(num_pos)

        tmp_list = [0] * len(labels)
        
        for fine_label in labels:
            if fine_label != label:
                
                num = int(np.sum(preds[pos] == fine_label))
                tmp_list[label2id[fine_label]] = num
                
        results_fine[label] = tmp_list

    return results, results_fine

def TSNE_reduce_feats(feats, dim):
    estimator = TSNE(n_components=dim)
    # estimator = TSNE(n_components=2, n_iter=8000, learning_rate=10, n_iter_without_progress=1200)
    print(feats.shape)
    print(feats.shape)
    reduce_feats = estimator.fit_transform(feats)
    
    return reduce_feats

def discover_centers(args, data, outputs):
    
    model_dir, output_file_dir, _ = set_path(args)

    predictions = list([data.all_label_list[idx] for idx in outputs[0]]) 
    true_labels = list([data.all_label_list[idx] for idx in outputs[1]]) 
    feats = outputs[2]
    reduce_feats = TSNE_reduce_feats(feats, 2)

    reduce_centers = []
    labels = np.unique(outputs[1])
    for label in labels:
        print(data.all_label_list[label])
        pos = list(np.where(outputs[1] == label)[0])
        center = np.mean(reduce_feats[pos], axis = 0)
        print('center', center)
        center = [round(float(x), 2) for x in center]
        reduce_centers.append(center)

    print(reduce_centers)
    # reduce_centers = [round(x, 2) for x in reduce_centers]
    all_dict = {}
    static_dir = os.path.join(args.frontend_dir, args.type)
    draw_center_r_path = os.path.join(static_dir, args.method + '_analysis.json') 
    
    known_centers = []
    open_centers = []
    for idx, center in enumerate(reduce_centers):
        label = data.all_label_list[idx]
        if label in data.known_label_list:
            point = center + [label]
            known_centers.append(point)
        else:
            point = center + [label]
            open_centers.append(point)

    center_dict = {}
    center_dict['Known Intent Centers'] = known_centers
    center_dict['Open Intent Centers'] = open_centers
    name = args.method + '_' +  args.dataset 
    all_dict[name] = center_dict
    json_add(all_dict, draw_center_r_path)

####################################Unsupervised Utils#######################################################
import re
from collections import defaultdict
import logging
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import seaborn as sns
import tensorflow as tf
import random as rn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Modeling
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.optimizers import SGD
from keras.engine.topology import Layer, InputSpec
# Evaluation
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder

# Visualization
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud

def check_inputs(args):
    check_list_labeled_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    check_list_known_cls_ratios = [0.25, 0.5, 0.75] 
    if args.labeled_ratio not in check_list_labeled_ratios:
        print('The assigned labeled ratio is unavailable!')
        return False
    if args.known_cls_ratio not in check_list_known_cls_ratios:
        print('The assigned known class ratio is unavailable!')
        return False

    return True

def get_manager(args, data):

    if not os.path.exists(os.path.join(args.train_data_dir, args.type)):
        os.makedirs(os.path.join(args.train_data_dir, args.type))

    module_names = [args.type, 'methods', args.setting, args.method, 'manager']
    import_name = ".".join(module_names)
    method = importlib.import_module(import_name) 
    manager = method.ModelManager(args, data)   

    return manager

def set_path(args):
    
    concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
    output_file_name = "_".join([str(x) for x in concat_names])
    output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
    output_file_dir = os.path.join(output_dir, args.save_results_path)
    model_dir = os.path.join(output_dir, args.model_dir)
    pretrain_model_dir = os.path.join(output_dir, 'pretrain')

    for di in [model_dir, output_file_dir, pretrain_model_dir]:
        if not os.path.exists(di):
            os.makedirs(di)

    return model_dir, output_file_dir, pretrain_model_dir

def load_pretrained_model(model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    classifier_params = ['classifier.weight','classifier.bias']
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def freeze_bert_parameters(model):
    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model

def save_discover_backend_results(manager, args, data):
    if not os.path.exists(manager.output_file_dir):
        os.makedirs(manager.output_file_dir)
    np.save(os.path.join(manager.output_file_dir, 'labels.npy'), data.all_label_list)

    var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed]
    names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(manager.test_results,**vars_dict)
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

def save_discover_frontend_results(args, data, outputs):

    results_path = os.path.join(args.train_data_dir, args.type, 'results.csv')

    static_dir = os.path.join(args.frontend_dir, args.type)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    ind, _ = hungray_aligment(outputs[1], outputs[0])
    map_ = {i[0]:i[1] for i in ind}
    y_pred = np.array([map_[idx] for idx in outputs[0]])

    #save true_false predictions
    predictions = list([data.all_label_list[idx] for idx in y_pred]) 
    true_labels = list([data.all_label_list[idx] for idx in outputs[1]]) 
    # predictions = list([data.known_label_list[idx] for idx in outputs[0]]) 
    # true_labels = list([data.known_label_list[idx] for idx in outputs[1]]) 
    predict_t_f, predict_t_f_fine = cal_true_false(true_labels, predictions)
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

def keywords_extraction(args, data, outputs):

    test_label_ids, test_true_ids, feats = outputs[0], outputs[1], outputs[2]
    test_examples = np.array([example.text_a for example in data.test_examples])
    keywords_model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = []

    intent_label_list_unique = np.unique(test_label_ids)
    dataset_info = {}
    dataset_info_json_path = os.path.join(sys.path[0], '../frontend/static/jsons/open_intent_discovery', 'analysis_table_info.json')
    if os.path.exists(dataset_info_json_path):
        with open(dataset_info_json_path, 'r') as load_f:
            dataset_info = json.load(load_f)
    class_list = []

    for intent_label_item in intent_label_list_unique:
        pos = list( np.where(test_label_ids == intent_label_item)[0] )
        cluster_texts = test_examples[pos]
        
        doc = " ".join(cluster_texts)
        keywords_cluster = keywords_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n = 3)
        strs_class_name = []
        for keyword_item in keywords_cluster:
            strs_class_name_tmp = '(' + str(keyword_item[0]) + ', ' + str( '%.2f' % (keyword_item[1]*100) ) + '%)'
            strs_class_name.append(strs_class_name_tmp)
        class_item = ", ".join(strs_class_name)
        # print('pipe.py-63:'+'\n'*3, class_item)
        class_list.append({"label_name": class_item, "label_text_num":len(cluster_texts),
            "dataset_name":args.dataset, "method": args.method,"class_type":'open'})
        text_list = []
        for sent in cluster_texts:
            keywords_sent = keywords_model.extract_keywords(sent, keyphrase_ngram_range=(1,2), top_n = 3)
            keywords_sent_len = len(keywords_sent)
            can_1,can_2,can_3 = 'None', 'None', 'None'
            conf_1,conf_2,conf_3 = '0', '0', '0'
            try:
                if keywords_sent_len == 0:
                    can_1 = keywords_cluster[0][0]
                    conf_1 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                    can_2 = keywords_cluster[1][0]
                    conf_2 = '%.2f' % (keywords_cluster[1][1]*100) + '%'
                    can_3 = keywords_cluster[2][0]
                    conf_3 = '%.2f' % (keywords_cluster[2][1]*100) + '%'
                elif keywords_sent_len == 1:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_cluster[0][0]
                    conf_2 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                    can_3 = keywords_cluster[1][0]
                    conf_3 = '%.2f' % (keywords_cluster[1][1]*100) + '%'
                elif keywords_sent_len == 2:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_sent[1][0]
                    conf_2 = '%.2f' % (keywords_sent[1][1]*100) + '%'
                    can_3 = keywords_cluster[0][0]
                    conf_3 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                elif keywords_sent_len == 3:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_sent[1][0]
                    conf_2 = '%.2f' % (keywords_sent[1][1]*100) + '%'
                    can_3 = keywords_sent[2][0]
                    conf_3 = '%.2f' % (keywords_sent[2][1]*100) + '%'
            except :
                print('run_discover.py:\t104:\tthere has an error')

            text_list.append({"dataset_name":args.dataset, "class_type":'open',
                "label_name": class_item,
                "method": args.method,
                "can_1": can_1,"can_2": can_2,"can_3":can_3,
                "conf_1": conf_1,"conf_2": conf_2,"conf_3": conf_3,
                "text": sent
            })
        dataset_info['text_list_'+args.dataset+"_"+args.method+"_open_"+class_item] = text_list
    dataset_info["class_list_"+args.dataset+"_"+args.method+"_open"] = class_list
    dataset_id_in_dataset_list = -1
    json_add(dataset_info, dataset_info_json_path)


# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def set_allow_growth(device):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list = device
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess) # set this TensorFlow session as the default session for Keras


def create_logger(app_name="root", level=logging.DEBUG):
    # 基礎設定
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler('logs/' + app_name + '.log', 'w', 'utf-8'), ])

    # 定義 handler 輸出 sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)

    # handler 設定輸出格式
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(app_name)
    return logger


def plot_cluster_pie(results, k, n_cols, df_plot, y_pred):
    NMI = results['NMI']
    ACC = results['ACC']
    n_rows = int(np.ceil(k/n_cols))

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/k) for i in range(k)]

    d = {val:key for key, val in enumerate(df_plot.label.unique())}
    df_plot['y_pred'] = y_pred
    fig, axes = plt.subplots(nrows=n_rows,  ncols=n_cols, figsize=(round(3.5*n_cols), 4*n_rows))
    for i in tqdm(range(k)):
        x = i//n_cols
        y = i%n_cols

        df_ = df_plot[df_plot.y_pred==i]
        y_true_vc = df_.label.value_counts()
        if y_true_vc.size==0:
            print("Encounter empty cluster, ignore")
            continue
            
        explode = [0] * len(y_true_vc.values)
        explode[0] = 0.1
        labels = y_true_vc.index
        cs = [colors[d[label]] for label in labels]
        axes[x, y].pie(y_true_vc.values, explode=explode, labels=labels, colors=cs)

        label_majority = y_true_vc.index[0]
        percentage = y_true_vc.values[0]/sum(y_true_vc) * 100
        n_samples = df_.shape[0]
        title = "%d: #=%d, %s(%1.1f%%)" % (i, n_samples, label_majority,  percentage)
        axes[x, y].set_title(title)

        centre_circle = plt.Circle((0,0), 0.50, fc='white')
        axes[x, y].add_artist(centre_circle)
        fig.tight_layout()
    fig.suptitle("K=%d (ACC=%1.2f, NMI=%1.2f)" % (k, ACC, NMI), fontsize=24)
    fig.subplots_adjust(top=0.9)

def plot_cluster_wordcloud(results, k, n_cols, df_plot, y_pred):
    NMI = results['NMI']
    ACC = results['ACC']
    n_rows = int(np.ceil(k/n_cols))
    stop_words = set(stopwords.words('english') + ['.', '\'', ','])
    fig, axes = plt.subplots(nrows=n_rows,  ncols=n_cols, figsize=(round(3.5*n_cols), 4*n_rows))
    for i in tqdm(range(k)):
        x = i//n_cols
        y = i%n_cols
        cnt = Counter()
        df_ = df_plot[df_plot.y_pred==i]
        y_true_vc = df_.label.value_counts()
        if y_true_vc.size==0: # 
            print("Encounter empty cluster, ignore")
            continue
            
        label_majority = y_true_vc.index[0]
        percentage = y_true_vc.values[0]/sum(y_true_vc) * 100

        for sentence in df_['words'].tolist():
            for word in sentence:
                if word not in stop_words:
                    cnt[word] += 1
        wordcloud = WordCloud(width=400, height=400, relative_scaling=0.5, normalize_plurals=False, 
                              background_color='white'
                             ).generate_from_frequencies(cnt)
        n_samples = df_.shape[0]
        title = "%d: #=%d, %s(%1.1f%%)" % (i, n_samples, label_majority,  percentage)
        axes[x, y].imshow(wordcloud, interpolation='bilinear')
        axes[x, y].set_title(title)

        centre_circle = plt.Circle((0,0),0.50, fc='white')
        axes[x, y].add_artist(centre_circle)
        axes[x, y].axis("off")
        fig.tight_layout()
    fig.suptitle("K=%d (ACC=%1.2f, NMI=%1.2f)" % (k, ACC, NMI), fontsize=24)
    fig.subplots_adjust(top=0.9)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12,10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('result.png')
    


