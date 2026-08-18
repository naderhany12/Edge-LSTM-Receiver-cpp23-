import numpy as np
import tensorflow as tf


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snr_db , **kwargs):
        super(AWGNChannel, self).__init__(**kwargs)

        snr_linear = 10 ** (snr_db/10)

        self.noise_std = np.sqrt(1.0/(2.0 * snr_linear))

    def call(self, inputs):
        # Add Gaussian noise to the input
        noise = tf.random.normal(shape=tf.shape(inputs), stddev=self.noise_std)
        noisy_inputs = inputs + noise
        return noisy_inputs

class Transmitter(tf.keras.layers.Layer):
    def __init__(self, feature_dim ,**kwargs):
        super(Transmitter, self).__init__(**kwargs)

        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(feature_dim, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        encoded = self.dense2(x)
        mean_energy = tf.reduce_mean(tf.square(encoded), axis = -1 , keepdims = True)
        normalized_signal = encoded / tf.sqrt(mean_energy + 1e-7)  ##Normalize the signal to have unit energy
        return normalized_signal

class Receiver(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, feature_dim, **kwargs):
        super(Receiver, self).__init__(**kwargs)

        self.lstm = tf.keras.layers.LSTM(hidden_dim , return_sequences=True)
        self.dense = tf.keras.layers.Dense(feature_dim)

    def call(self , inputs):
        x = self.lstm(inputs)
        logits = self.dense(x)
        return logits

class Autoencoder(tf.keras.Model):
    def __init__(self , feature_dim , hidden_dim , snr_db , **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

        self.tx = Transmitter(feature_dim)
        self.channel = AWGNChannel(snr_db)
        self.rx = Receiver(hidden_dim, feature_dim)

    def call(self, inputs):
        encoded_signal = self.tx(inputs)
        noisy_signal = self.channel(encoded_signal)
        decoded_signal = self.rx(noisy_signal)
        return decoded_signal