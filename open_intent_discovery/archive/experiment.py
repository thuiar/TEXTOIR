
from utils_unsupervised import *
from model_unsupervised import *
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import nltk
from sklearn.metrics import confusion_matrix
from datetime import datetime
from keras.utils import plot_model
from keras.optimizers import SGD
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import random
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='banking')
parser.add_argument('--fraction', type=int, default=1)
parser.add_argument('--labeled_ratio', type=float, default=0.1)
parser.add_argument('--known_cls_ratio', type=float, default=0.75)
parser.add_argument('--seed', type=int, default=44)
parser.add_argument('--gpu_id', type=str, default='0')

args = parser.parse_args()
task_name = dataset = args.task_name
fraction = args.fraction
labeled_ratio = args.labeled_ratio
seed = args.seed

# Data preprocessing
MAX_SEQ_LEN = None
MAX_NUM_WORDS = 10000

# Modeling DEC & DCN
maxiter = 12000
batch_size = 256
update_interval = 400
tol = 0.001 # tolerance threshold to stop training
number = "0"
fraction = args.fraction

# #### Sentence embedding (MeanPooling)
# print("Building GloVe (D=300)...")
# base_dir = '/home/sharing/sharing/pretrained_embedding/glove/'
# embedding_matrix, embeddings_index = get_glove(base_dir, MAX_FEATURES, word_index)
# gev = GloVeEmbeddingVectorizer(embedding_matrix, index_word, X_train)
# emb_train = gev.transform(X_train, method='mean')
# emb_test = gev.transform(X_test, method='mean')
# # emb_idf_train = gev.transform(X_train, method='idf')
# # emb_idf_test = gev.transform(X_test, method='idf')

# train_csv = pd.read_csv('../../data/banking/train.tsv', sep = '\t')
# dev_csv = pd.read_csv('../../data/banking/dev.tsv', sep = '\t')
# test_csv = pd.read_csv('../../data/banking/test.tsv', sep = '\t')

# train_data_list = [[x, y] for x, y in zip(train_csv['text'], train_csv['label'])]
# dev_data_list = [[x, y] for x, y in zip(dev_csv['text'], dev_csv['label'])]
# test_data_list = [[x, y] for x, y in zip(test_csv['text'], test_csv['label'])]

# all_data_list = train_data_list + dev_data_list + test_data_list
# all_data_frame = pd.DataFrame(all_data_list, columns = ['text', 'label'])

# train_data_list = train_data_list + dev_data_list
# train_data_frame = pd.DataFrame(train_data_list, columns = ['text', 'label'])
# # dev_data_frame = pd.DataFrame(dev_data_list, columns = ['text', 'label'])
# test_data_frame = pd.DataFrame(test_data_list, columns = ['text', 'label'])

# df = all_data_frame
df = pd.read_csv('../../data/banking/train.tsv', sep='\t')
set_allow_growth(args.gpu_id)
df['words'] = df['text'].apply(word_tokenize)
le = LabelEncoder() 
df['y_true'] = le.fit_transform(df['label'])
df['text'] = df['words'].apply(lambda l: " ".join(l))
texts = df['words'].tolist() 

# filters without "," and "."
tk = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~') 
tk.fit_on_texts(texts)

# Fix UNK problem
tk.word_index = {e:i for e,i in tk.word_index.items() if i <= MAX_NUM_WORDS} # <= because tokenizer is 1 indexed
tk.word_index[tk.oov_token] = MAX_NUM_WORDS+1

word_index = tk.word_index
index_word = {v: k for k, v in word_index.items()}
MAX_FEATURES = min(MAX_NUM_WORDS+1, len(word_index)) + 1
sequences = tk.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

df_train, df_test = train_test_split(df, test_size=labeled_ratio, stratify=df.label, shuffle=True, random_state=seed)
X_train = sequences_pad[df_train.index]
X_test = sequences_pad[df_test.index]

# y_train = df_train.y_true.values
# y_test = df_test.y_true.values
# df_train = train_data_frame
# df_test = test_data_frame
# y_test = all_data_frame.y_true.values[test_data_frame.index]

print("Building TFIDF...")
vec_tfidf = TfidfVectorizer(max_features=2000)
tfidf_train = vec_tfidf.fit_transform(df_train['text'].tolist()).todense()
tfidf_test = vec_tfidf.transform(df_test['text'].tolist()).todense()

print("Training: SAE(emb)")
sae_emb = get_stacked_autoencoder(tfidf_train.shape[1])

sae_emb.fit(tfidf_train, tfidf_train, epochs=5000, batch_size=4096, shuffle=True, 
			validation_data=(tfidf_test, tfidf_test), verbose=1)
sae_emb.save_weights('../SAE_' + dataset + '_' + number + '.h5')
# sae_emb.load_weights('/home/sharing/disk2/zhanghanlei/save_data_162/TEXTOIR/outputs/open_intent_discovery/SAE-KM_banking_0.75_0.1_sae_0')
emb_train_sae = get_encoded(sae_emb, [tfidf_train], 3)
emb_test_sae = get_encoded(sae_emb, [tfidf_test], 3)

#### Sentence embedding (MeanPooling)
# print("Building GloVe (D=300)...")
# base_dir = '/home/sharing/sharing/pretrained_embedding/glove/'
# embedding_matrix, embeddings_index = get_glove(base_dir, MAX_FEATURES, word_index)
# gev = GloVeEmbeddingVectorizer(embedding_matrix, index_word, X_train)
# emb_train = gev.transform(X_train, method='mean')
# emb_test = gev.transform(X_test, method='mean')
# emb_idf_train = gev.transform(X_train, method='idf')
# emb_idf_test = gev.transform(X_test, method='idf')

# # #### Baseline: GloVe(mean) + KM/AG
results_all = {}
k = 58
# df_plot = df_test

# km = KMeans(n_clusters=k, n_jobs=-1)
# km.fit(emb_train)
# y_pred = km.predict(emb_test)
# results = clustering_score(y_test, y_pred)
# results_all.update({'KM': results})

# ### Baseline: GloVe(mean) + AG
# ag = AgglomerativeClustering(n_clusters=k)
# ag.fit(emb_test)
# results = clustering_score(y_test, ag.labels_)
# results_all.update({'AG': results})


#### Baseline: SAE + KM
df_plot = df_test
km = KMeans(n_clusters=k, n_jobs=-1, random_state=seed)
km.fit(emb_train_sae)
y_pred = km.predict(emb_test_sae)
results = clustering_score(y_test, y_pred)
results_all.update({'SAE-KM': results})
print(results_all)

# ########DEC##############
# df_plot = df_test
# loss = 0
# index = 0
# index_array = np.arange(tfidf_train.shape[0])
# x = tfidf_train
# y = y_train

# # Initialize model
# sae_emb.load_weights('../datasets/SAE_' + dataset + '_' + number + '.h5')

# clustering_layer = ClusteringLayer(k,name='clustering')(sae_emb.layers[3].output)
# model = Model(inputs=sae_emb.input, outputs=clustering_layer)
# model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
# # plot_model(model, to_file='output/model.png', show_shapes=True)

# # Initialize cluster centers using k-means
# kmeans = KMeans(n_clusters=k, n_init=20, n_jobs=-1)
# y_pred = kmeans.fit_predict(emb_train_sae)
# y_pred_last = np.copy(y_pred)
# model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# for ite in range(int(maxiter)):
# 	if ite % update_interval == 0:
# 		q = model.predict(x, verbose=0)
# 		p = target_distribution(q)  # update the auxiliary target distribution p
# 		# evaluate the clustering performance
# 		y_pred = q.argmax(1)
# 		if y is not None:
# 			results = clustering_score(y, y_pred)
# 			print('Iter=', ite, results, 'loss=', np.round(loss, 5))
# 		# check stop criterion - model convergence
# 		delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
# 		y_pred_last = np.copy(y_pred)
# 		if ite > 0 and delta_label < tol:
# 			print('delta_label ', delta_label, '< tol ', tol)
# 			print('Reached tolerance threshold. Stopping training.')
# 			break
# 	idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
# 	loss = model.train_on_batch(x=x[idx], y=p[idx])
# 	index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

# # model.save_weights('data/DEC_' + dataset + '_' + number +'.h5')

# # Evaluation
# q = model.predict(tfidf_test, verbose=0)
# p = target_distribution(q)  # update the auxiliary target distribution p
# y_pred = q.argmax(1)
# results = clustering_score(y_test, y_pred)
# results['k_pred'] = len(set(y_pred))
# results_all.update({'DEC': results})


# #### Baseline: DCN
# df_plot = df_test
# loss = 0
# index = 0
# index_array = np.arange(tfidf_train.shape[0])
# x = tfidf_train
# y = y_train

# # Initialize model
# del model
# sae_emb.load_weights('../datasets/SAE_' + dataset + '_' + number + '.h5')
# clustering_layer = ClusteringLayer(k, name='clustering')(sae_emb.layers[3].output)
# model = Model(inputs=sae_emb.input, outputs=[clustering_layer, sae_emb.output])
# model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=SGD(0.01, 0.9))
# plot_model(model, to_file='output/model.png', show_shapes=True)

# # Initialize cluster centers using k-means
# kmeans = KMeans(n_clusters=k, n_init=20, n_jobs=-1)
# y_pred = kmeans.fit_predict(emb_train_sae)
# y_pred_last = np.copy(y_pred)
# model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# for ite in range(int(maxiter)):
# 	if ite % update_interval == 0:
# 		q, _ = model.predict(x, verbose=0)
# 		p = target_distribution(q)  # update the auxiliary target distribution p
# 		# evaluate the clustering performance
# 		y_pred = q.argmax(1)
# 		if y is not None:
# 			results = clustering_score(y, y_pred)
# 			print('Iter=', ite, results, 'loss=', np.round(loss, 5))
# 		# check stop criterion - model convergence
# 		delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
# 		y_pred_last = np.copy(y_pred)
# 		if ite > 0 and delta_label < tol:
# 			print('delta_label ', delta_label, '< tol ', tol)
# 			print('Reached tolerance threshold. Stopping training.')
# 			break
# 	idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
# 	loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
# 	index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

# # # model.save_weights('data/DCN_' + dataset + '_' + number + '.h5')

# # Evaluation
# q, _ = model.predict(tfidf_test, verbose=0)
# y_pred = q.argmax(1)
# results = clustering_score(y_test, y_pred)
# results['k_pred'] = len(set(y_pred))
# results_all.update({'DCN': results})

# ############write into csv################
# baselines = ['KM','AG','SAE-KM','DEC','DCN']
# var = [task_name, labeled_ratio, seed,fraction]
# names = ['task_name','labeled_ratio','seed','fraction']
# vars_dict = {k:v for k,v in zip(names, var) }
# file_path = 'results.csv'
# save_results_path = 'output'
# #results_path = os.path.join(save_results_path,file_path)
# for baseline in baselines:
# 	each_results_path = os.path.join(save_results_path,baseline,file_path) 
# 	if results_all.get(baseline) != None:
# 		results_ = dict(results_all.get(baseline),**vars_dict)
# 		keys = list(results_.keys())
# 		values = list(results_.values())
# 		if not os.path.exists(each_results_path):
# 			ori = []
# 			ori.append(values)
# 			df1 = pd.DataFrame(ori,columns = keys)
# 			df1.to_csv(each_results_path,index=False)
# 		else:
# 			df1 = pd.read_csv(each_results_path)
# 			new = pd.DataFrame(results_,index=[1])
# 			df1 = df1.append(new,ignore_index=True)
# 			df1.to_csv(each_results_path,index=False)
# print(results_all)
