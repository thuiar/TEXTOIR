import pandas as pd
import os 
import numpy as np
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class UNSUP_Loader:

    def __init__(self, args, base_attrs):
        
        self.all_data, self.train_data, self.test_data = self.get_examples(base_attrs)
        self.all_data['words'] = self.all_data['text'].apply(word_tokenize)
        le = LabelEncoder()
        self.all_data['y_true'] = le.fit_transform(self.all_data['label'])
        self.all_data['text'] = self.all_data['words'].apply(lambda l: " ".join(l))
        self.train_data, self.test_data = self.all_data.iloc[self.train_data.index], self.all_data.iloc[self.test_data.index]
        self.test_true_labels = self.all_data.y_true.values[self.test_data.index]
        
        if args.backbone == 'glove':
            self.embedding_matrix, self.index_word, self.train_data, self.test_data = \
                get_glove_data(args, self.all_data, self.train_data, self.test_data)
        elif args.backbone == 'sae':
            self.tfidf_train, self.tfidf_test = get_tfidf_data(args, self.train_data, self.test_data)
    
        # self.train_data['words'] = self.train_data['text'].apply(word_tokenize)
        # self.train_texts = self.train_data['words'].tolist()
        # self.train_pad = self.get_sequences_pad(self.train_texts, args)
        # self.train_x = self.train_pad[self.train_data.index]
        
        # self.test_data['words'] = self.test_data['text'].apply(word_tokenize)
        # self.test_texts = self.test_data['words'].tolist()
        # self.test_pad = self.get_sequences_pad(self.test_texts, args)
        # self.le = LabelEncoder()
        # self.test_data['y_true'] = self.le.fit_transform(self.test_data['label'])
        # self.test_x = self.train_pad[self.test_data.index]
        # self.test_y = self.test_data.y_true.values
        # # self.all_data, self.train_data, self.dev_data, self.test_data = self.get_examples(base_attrs)
        
        
        
        # # self.le = LabelEncoder()
        # # self.train_data['y_true'] = self.le.fit_transform(self.train_data['label'])
        # # self.test_data['y_true'] = self.le.fit_transform(self.test_data['label'])

        # # df_train = train_test_split(self.train_data, test_size=0, stratify=self.train_data.label, shuffle=True, random_state=args.seed)
        # # # df_test = train_test_split(self.test_data, test_size=0, stratify=self.test_data.label, shuffle=True, random_state=args.seed)
        # df_train, df_test = train_test_split(self.all_data, test_size=0.2, stratify=self.all_data.label, shuffle=True, random_state=args.seed)
        # df_test, df_t = train_test_split(self.all_data, test_size=0.9, stratify=self.all_data.label, shuffle=True, random_state=args.seed)
        # # print(df_test)
        # print(self.test_data)
        
        
        # # self.train_x = self.sequences_pad[self.all_data.index]
        # # self.test_x = self.sequences_pad[self.test_data.index]
        # # self.le = LabelEncoder()
        # # self.test_data['y_true'] = self.le.fit_transform(self.test_data['label'])
        # # print('11111111111', set(self.test_data['y_true']))
        # # self.test_y = self.test_data.y_true.values


    def get_examples(self, base_attrs):
        
        train_csv = pd.read_csv(os.path.join(base_attrs['data_dir'],'train.tsv'), sep = '\t')
        dev_csv = pd.read_csv(os.path.join(base_attrs['data_dir'],'dev.tsv'), sep = '\t')
        test_csv = pd.read_csv(os.path.join(base_attrs['data_dir'],'test.tsv'), sep = '\t')

        train_data_list = [[x, y] for x, y in zip(train_csv['text'], train_csv['label'])]
        dev_data_list = [[x, y] for x, y in zip(dev_csv['text'], dev_csv['label'])]
        test_data_list = [[x, y] for x, y in zip(test_csv['text'], test_csv['label'])]

        all_data_list = train_data_list + dev_data_list + test_data_list
        all_data_frame = pd.DataFrame(all_data_list, columns = ['text', 'label'])

        train_data_list = train_data_list + dev_data_list
        train_data_frame = pd.DataFrame(train_data_list, columns = ['text', 'label'])
        # dev_data_frame = pd.DataFrame(dev_data_list, columns = ['text', 'label'])
        test_data_frame = pd.DataFrame(test_data_list, columns = ['text', 'label'])

        return all_data_frame, train_data_frame, test_data_frame

def get_tfidf_data(args, train_data, test_data):

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_tfidf = TfidfVectorizer(max_features=args.feat_dim)
    tfidf_train = vec_tfidf.fit_transform(train_data['text'].tolist()).todense()
    tfidf_test = vec_tfidf.transform(test_data['text'].tolist()).todense()

    return tfidf_train, tfidf_test

def get_glove_data(args, all_data, train_data, test_data):

    texts = all_data['words'].tolist()
    
    tokenizer = Tokenizer(num_words = args.max_num_words, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')
    tokenizer.fit_on_texts(texts)
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= args.max_num_words} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = args.max_num_words + 1# because tokenizer is 1 indexed

    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    max_features = min(args.max_num_words + 1, len(word_index)) + 1
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(sequences, maxlen = args.max_seq_length, padding='post', truncating='post')

    train_x = sequences_pad[train_data.index]
    test_x = sequences_pad[test_data.index]
    
    
    embedding_matrix, embeddings_index = get_glove_embedding(args, max_features, word_index)

    return embedding_matrix, index_word, train_x, test_x

def get_glove_embedding(args, MAX_FEATURES, word_index):

    EMBEDDING_DIM = 300
    EMBEDDING_FILE = os.path.join(args.glove_model, 'glove.6B.' + str(EMBEDDING_DIM) +'d.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    #read token embedding and process, form a dict (one word -> one vector)
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,encoding="utf-8"))
    #get value from dict
    all_embs = np.stack(embeddings_index.values())
    #cal mean and std
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    """Guassian distribution
    """
    # pad zero to none 10002, 300
    embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES+1, EMBEDDING_DIM))
    #
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    #embedding_matrix (MAX_FEATURES, ) random initialization for unmarked token
    return embedding_matrix, embeddings_index


