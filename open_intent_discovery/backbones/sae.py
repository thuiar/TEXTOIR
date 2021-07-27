
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

def get_encoded(model, data, nb_layer):
    transform = K.function([model.layers[0].input], 
                           [model.layers[nb_layer].output])
    return transform(data)[0]

def get_sae(args, sae_emb, tfidf_train, tfidf_test):
    
    emb_train_sae = get_encoded(sae_emb, [tfidf_train], 3)
    emb_test_sae = get_encoded(sae_emb, [tfidf_test], 3)
    
    return emb_train_sae, emb_test_sae

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