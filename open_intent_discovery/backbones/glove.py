import numpy as np
import os
from collections import defaultdict

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

