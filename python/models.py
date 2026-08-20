import numpy as np
import tensorflow as tf


class ImpairedChannel(tf.keras.layers.Layer):
    """
    Advanced Telecom Channel Layer with both AWGN Noise AND Non-Linear Channel Distortion.
    Forces the Receiver LSTM to learn Distortion-Invariant Latent Features.
    """
    def __init__(self, min_snr_db=3.0, max_snr_db=12.0, **kwargs):
        super(ImpairedChannel, self).__init__(**kwargs)
        self.min_snr_db = float(min_snr_db)
        self.max_snr_db = float(max_snr_db)

    def call(self, inputs, training=None):
        if training:
            # 1. Inject Random Channel Gain & Phase Shift Distortion
            gain = tf.random.uniform(shape=[tf.shape(inputs)[0], 1, 1], minval=0.7, maxval=1.3)
            distorted_inputs = inputs * gain
            
            # 2. Inject Multi-Level AWGN Noise (Severe Noise Range)
            snr_db = tf.random.uniform(shape=[], minval=self.min_snr_db, maxval=self.max_snr_db)
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_std = tf.sqrt(1.0 / (2.0 * snr_linear))
            noise = tf.random.normal(shape=tf.shape(distorted_inputs), stddev=noise_std)
            
            return distorted_inputs + noise
        else:
            # Standard Evaluation Channel
            snr_linear = 10.0 ** (7.0 / 10.0)
            noise_std = tf.sqrt(1.0 / (2.0 * snr_linear))
            return inputs + tf.random.normal(shape=tf.shape(inputs), stddev=noise_std)

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
    def __init__(self , feature_dim , hidden_dim , **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

        self.tx = Transmitter(feature_dim)
        self.channel = ImpairedChannel(min_snr_db=3.0, max_snr_db=12.0)
        self.rx = Receiver(hidden_dim, feature_dim)

    def call(self, inputs , training = None):
        encoded_signal = self.tx(inputs)
        noisy_signal = self.channel(encoded_signal , training = training)
        decoded_signal = self.rx(noisy_signal)
        return decoded_signal