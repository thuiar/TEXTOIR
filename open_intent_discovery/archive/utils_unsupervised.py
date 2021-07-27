# -*- coding:utf-8 -*-
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import logging
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors
# from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random as rn
import torch
from torch import nn, optim
from torch.nn import functional as F

# Evaluation
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud



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


def load_single(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("../data/" + dataset + "/" + partition + ".seq.in", encoding="utf-8") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("../data/" + dataset + "/" + partition + ".label", encoding="utf-8") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row


def get_glove(base_dir, MAX_FEATURES, word_index):
    EMBEDDING_DIM = 300
    EMBEDDING_FILE = os.path.join(base_dir, 'glove.6B.' + str(EMBEDDING_DIM) +'d.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    #将词嵌入读出来，去掉换行符，按空格分开形成列表，再把整体变成一个字典，一个词对应一个向量
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,encoding="utf-8"))
    #取出字典的value值变成一个列表
    all_embs = np.stack(embeddings_index.values())
    #计算所有值的平均值和标准差
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    """按照词向量均值和标准差拟合正态分布
    """
    # embedding_matrix的长度多一行，不存在embedding的词的值都为0 (pad)  10002,300
    embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES+1, EMBEDDING_DIM))
    #
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    #embedding_matrix对应前MAX_FEATURES个词，每个词的词向量，如果是未登录词就是随机初始化的
    return embedding_matrix, embeddings_index


class GloVeEmbeddingVectorizer(object):
    def __init__(self, embedding_matrix, index_word, X=None):
        self.embedding_matrix = embedding_matrix
        self.dim = embedding_matrix.shape[1]
        if X is not None:
            self.index_word = index_word
            self.D = embedding_matrix.shape[0]
            self.idf = self.get_idf(X)
        
    def get_idf(self, X):
        d = defaultdict(int)
        idf = defaultdict(int)
        if isinstance(X,list):
            for e in X:
                for word_indices in e:
                    for idx in word_indices:
                        d[idx] += 1
        else:
            for word_indices in X:
                for idx in word_indices:
                    d[idx]+= 1
        idf = {k:np.log(self.D/v) for k, v in d.items()}
        return idf
    
    def transform(self, X, method='mean'):
        sentence_embs = []
        for word_indices in X:
            word_embs = []
            dividend = 0
            for idx in word_indices:
                if idx in self.index_word and idx!=0:
                    if method=='mean':
                        weight = 1
                    elif method=='idf':
                        mark = self.idf.get(idx,None)
                        if mark is not None:
                            weight = self.idf[idx]
                        else:
                            weight = np.log(self.D / 1)
                       
                    word_embs.append(self.embedding_matrix[idx]*weight)
                    dividend += weight
            # no words founded in GloVe
            if dividend==0: 
                sentence_emb = np.zeros(self.dim)
            else:
                sentence_emb = np.divide(np.sum(word_embs, axis=0), dividend)
            sentence_embs.append(sentence_emb)
        return np.array(sentence_embs)


def hungray_aligment(y_true, y_pred):
    '''
    Find best alignment with Hungary algorithm, and return the index    
    '''
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size): # Confusion matrix
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)
    return ind, w
    
def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w
def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
#             'AMI': round(adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')*100),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}


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
    
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

