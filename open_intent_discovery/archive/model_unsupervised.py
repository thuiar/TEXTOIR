# Modeling
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer, InputSpec



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