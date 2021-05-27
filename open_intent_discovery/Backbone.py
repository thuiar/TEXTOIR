from .utils import *

###################################################################################################
def get_glove(base_dir, MAX_FEATURES, word_index):
    EMBEDDING_DIM = 300
    EMBEDDING_FILE = os.path.join(base_dir, 'glove.6B.' + str(EMBEDDING_DIM) +'d.txt')
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

#########################################################################################################
                    
def get_autoencoder(original_dim=300, encoding_dim=30):
    model = Sequential([Dense(encoding_dim, activation='relu', kernel_initializer='glorot_uniform', input_shape=(original_dim,)),
                        Dense(original_dim, kernel_initializer='glorot_uniform')])
    adam = Adam(lr=0.1, clipnorm=1)
    model.compile(optimizer='adam', loss='mse')
    return model

def get_stacked_autoencoder(original_dim=2000, encoding_dim=10):
    model = Sequential([Dense(500, activation='relu', kernel_initializer='glorot_uniform', input_shape=(original_dim,)),
                        Dense(500, activation='relu', kernel_initializer='glorot_uniform'),
                        Dense(2000, activation='relu', kernel_initializer='glorot_uniform'),
                        Dense(encoding_dim, activation='relu', kernel_initializer='glorot_uniform', name='encoded'),
                        Dense(2000, activation='relu', kernel_initializer='glorot_uniform'),
                        Dense(500, activation='relu', kernel_initializer='glorot_uniform'),
                        Dense(500, activation='relu', kernel_initializer='glorot_uniform'),
                        Dense(original_dim, kernel_initializer='glorot_uniform')])
    adam = Adam(lr=0.005, clipnorm=1)
    model.compile(optimizer='adam', loss='mse')
    return model


def get_encoded(model, data, nb_layer):
    transform = K.function([model.layers[0].input], 
                           [model.layers[nb_layer].output])
    return transform(data)[0]

# For DCN
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

################################################################
class bert(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(bert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = loss_fct(logits,labels)
                return loss
            else:
                return pooled_output, logits

################################################################
def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))

    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
        eps = 1e-7 
        assert predict.ndimension()==2,'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict += eps
        target += eps
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld
    
class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        kld = self.kld(prob1,prob2)
        output = self.hingeloss(kld,simi)
        return output

class BertForSimilarity(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForSimilarity,self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.normalization = nn.BatchNorm1d(config.hidden_size * 4)
        self.classifier = nn.Linear(config.hidden_size * 4, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self, input_ids, token_type_ids = None, attention_mask=None, labels=None, loss_fct=None, mode = None):
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, 
output_all_encoded_layers=False)
        encoded_layer_12 = encoded_layer_12.mean(dim = 1)
        feat1,feat2 = PairEnum(encoded_layer_12)
        feature_cat = torch.cat([feat1,feat2],1)
        pooled_output = self.dense(feature_cat)
        pooled_output = self.normalization(pooled_output)
        pooled_output = self.activation(pooled_output)
        logits = self.classifier(pooled_output)
        
        if mode == 'train':    
            loss = loss_fct(logits.view(-1,self.num_labels), labels.view(-1))
            return loss
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds

class KCLForBert(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(KCLForBert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, simi = None, mode = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = F.softmax(logits,dim=1)
        
        if mode == 'train':    
            flag = len(labels[labels != -1])
            if flag != 0:
                loss_ce = nn.CrossEntropyLoss()(logits[labels != -1], labels[labels != -1])
                
                prob1, prob2 = PairEnum(probs)
                criterion = KCL().cuda()
                loss_KCL = criterion(prob1, prob2, simi)
                loss = loss_ce + loss_KCL
            else:
                prob1,prob2 = PairEnum(probs)
                criterion = KCL().cuda()
                loss = criterion(prob1, prob2, simi)

            return loss
        else:
            preds = torch.argmax(probs, dim=1)
            return preds, pooled_output

######################################################################################################
class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
        
    def forward(self, prob1, prob2, simi=None):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return neglogP.mean()
        
class MCLForBert(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(MCLForBert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, simi = None, mode = None, ext_feats=False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        p = F.softmax(logits, dim = 1)
        if mode == 'train':
            flag = len(labels[labels != -1])
            
            if flag != 0:
                loss_ce = nn.CrossEntropyLoss()(logits[labels != -1], labels[labels != -1])

                p = F.softmax(logits, dim = 1)
                p1,p2 = PairEnum(p)

                simi = torch.matmul(p, p.transpose(0, -1)).view(-1)
                simi[simi > 0.5] = 1
                simi[simi < 0.5] = -1

                criterion = MCL().cuda()
                loss_MCL = criterion(p1, p2, simi)
                loss = loss_ce + loss_MCL
            else:
                p = F.softmax(logits, dim = 1)
                p1,p2 = PairEnum(p)

                simi = torch.matmul(p, p.transpose(0, -1)).view(-1)
                simi[simi > 0.5] = 1
                simi[simi < 0.5] = -1
                criterion = MCL().cuda()
                loss= criterion(p1, p2, simi)

            return loss
        else:
            pred_labels = torch.argmax(p,dim=1)
            if ext_feats:
                return pred_labels, pooled_output
            return pred_labels

#######################################################################################
class DTCForBert(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(DTCForBert, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        #train
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)
        
        #finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(num_labels, num_labels))

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None, loss_fct=None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            return pooled_output
        else:
            q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t() # Make sure each sample's n_values add up to 1.
            return logits, q       
    
    #########################CDAC+##########################
class BertForConstrainClustering(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForConstrainClustering, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        
        # train
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Pooling-mean
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        
        # finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(num_labels, num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                u_threshold=None, l_threshold=None, mode=None, labels=None, semi=False):
        eps = 1e-10
        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1)) # Pooling-mean
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if mode=='train':
            logits_norm = F.normalize(logits, p=2, dim=1)
            sim_mat = torch.matmul(logits_norm, logits_norm.transpose(0, -1))
            label_mat = labels.view(-1,1) - labels.view(1,-1)    
            label_mat[label_mat!=0] = -1 # dis-pair: label=-1
            label_mat[label_mat==0] = 1  # sim-pair: label=1
            label_mat[label_mat==-1] = 0 # dis-pair: label=0
            if not semi:
                pos_mask = (label_mat > u_threshold).type(torch.cuda.FloatTensor)
                neg_mask = (label_mat < l_threshold).type(torch.cuda.FloatTensor)
                pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                loss = (pos_entropy.mean() + neg_entropy.mean())*5
                return loss
            else:
                label_mat[labels==-1, :] = -1
                label_mat[:, labels==-1] = -1
                label_mat[label_mat==0] = 0
                label_mat[label_mat==1] = 1
                pos_mask = (sim_mat > u_threshold).type(torch.cuda.FloatTensor)
                neg_mask = (sim_mat < l_threshold).type(torch.cuda.FloatTensor)
                pos_mask[label_mat==1] = 1
                neg_mask[label_mat==0] = 1
                pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                loss = pos_entropy.mean() + neg_entropy.mean() + u_threshold - l_threshold
                return loss
        else:
            q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t() # Make sure each sample's n_values add up to 1.
            return logits, q