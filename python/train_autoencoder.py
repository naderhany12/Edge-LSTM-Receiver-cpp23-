import numpy as np
import tensorflow as tf
from export_weights import export_receiver_weights
from models import Autoencoder




#Traning hyperparameters
SEQUENCE_LEN = 10
FEATURE_DIM = 4
HIDDEN_DIM = 16
SNR_DB = 7.0
EPOCHS = 500
BATCH_SIZE = 64



def generate_data(num_samples):
    # Generate random sequences of shape (num_samples, SEQUENCE_LEN, FEATURE_DIM) 
    
    shape = (num_samples, SEQUENCE_LEN , FEATURE_DIM)
    data = np.random.randint(0 , 2 , size=shape).astype(np.float32)
    return data

def calculate_ber(y_true, y_pred):
    # Calculate bit error rate
    predections = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)
    errors = tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - predections))
    return errors

if __name__ == "__main__":
    data = generate_data(1000)
    model = Autoencoder(FEATURE_DIM, HIDDEN_DIM, SNR_DB)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    print("--- Training Autoencoder ---")
    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            logits = model(data)
            loss = loss_fn(data, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Test & Export
    test_data = generate_data(2000)
    test_logits = model(test_data)
    final_ber = calculate_ber(test_data, test_logits).numpy()
    print(f"Final Test BER: {final_ber * 100:.2f}%")

    if final_ber <= 0.01:
        export_receiver_weights(
            model, "../model_weights/lstm_signal_weights.bin"
        )