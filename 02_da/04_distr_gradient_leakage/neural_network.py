import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import gc


def encode_weights(weights):
    '''Encodes weight to be JSON seralizable'''
    weights_encoded = []
    for w in weights:
        weights_encoded.append(w.tolist())
    return weights_encoded

def decode_weights(weights):
    '''Decodes JSON-seralized weights'''
    weights_decoded = []
    for w in weights:
        weights_decoded.append(np.array(w))
    return weights_decoded

class NeuralNetwork:
    def _create_model(self):
        self.model = keras.Sequential([
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
        self.model.build(input_shape=(None,79))
        self.model.compile(loss="mse",
                           optimizer = tf.keras.optimizers.Adam(0.001))
        
    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                logical_gpus = tf.config.list_logical_devices('GPU')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        self._create_model()
        
    def train(self, train_features, train_labels,
              validation_split = 0.2, epochs = 100, batch_size = 10000):
        self.history = self.model.fit(train_features, train_labels, validation_split = validation_split,
                                      epochs = epochs, batch_size = batch_size, verbose=0)
        return self.history
        
    def predict(self, test_features):
        return self.model.predict(test_features, batch_size=100000, verbose=0)
    
    def reset(self):
        tf.keras.backend.clear_session()
        gc.collect()
        self._create_model()