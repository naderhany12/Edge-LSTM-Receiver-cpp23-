import os
import time
import numpy as np
import tensorflow as tf
from export_weights import export_receiver_weights
from models import Autoencoder




#Traning hyperparameters
SEQUENCE_LEN = 10
FEATURE_DIM = 4
HIDDEN_DIM = 16
SNR_DB = 7.0
EPOCHS = 400
SAMPLES_PER_EPOCH = 5000 # Dynamic dataset size per epoch
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
    model = Autoencoder(FEATURE_DIM, HIDDEN_DIM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(batch_data):
        with tf.GradientTape() as tape:
            logits = model(batch_data, training=True)
            loss = loss_fn(batch_data, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    os.makedirs("reports", exist_ok=True)
    report_file = open("reports/training_report.txt", "w", encoding="utf-8")
    
    print("--- Training Autoencoder (Graph Accelerated Mode) ---")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_data = generate_data(SAMPLES_PER_EPOCH)
        dataset = tf.data.Dataset.from_tensor_slices(epoch_data).batch(BATCH_SIZE)

        total_loss = 0.0
        steps = 0

        for batch_data in dataset:
            loss = train_step(batch_data)
            total_loss += loss.numpy()
            steps += 1

        avg_loss = total_loss / steps

        if epoch % 50 == 0 or epoch == 1: 
            eval_data = generate_data(1000)
            eval_logits = model(eval_data, training=False)
            ber = calculate_ber(eval_data, eval_logits).numpy()

            elapsed = time.time() - start_time
            log = f"Epoch {epoch:3d}/{EPOCHS} - Loss: {avg_loss:.4f} - Test BER: {ber * 100:.2f}% ({elapsed:.1f}s)"
            print(log)
            report_file.write(log + "\n")

    # Test & Export
    test_data = generate_data(4000)
    test_logits = model(test_data, training=False)
    final_ber = calculate_ber(test_data, test_logits).numpy()

    final_log = f"\nFinal Robust Base Model Test BER: {final_ber * 100:.3f}%"
    print(final_log)
    report_file.write(final_log + "\n")
    report_file.close()

    # Export Binary Weights for C++ Engine
    export_receiver_weights(model, "model_weights/lstm_signal_weights.bin")
    print("\n[SUCCESS] New Robust Model Weights exported to model_weights/lstm_signal_weights.bin")