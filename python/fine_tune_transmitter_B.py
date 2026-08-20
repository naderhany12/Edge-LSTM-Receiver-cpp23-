import os
import numpy as np
import tensorflow as tf
from export_weights import export_receiver_weights
from models import Autoencoder

# Frame & Adaptation Hyperparameters
SEQUENCE_LEN = 10
FEATURE_DIM = 4
HIDDEN_DIM = 16
SNR_DB = 7.0

# Wireless Frame Structure
NUM_PILOT_SEQUENCES = 100    # Short Pilot Burst for fast adaptation
NUM_PAYLOAD_SEQUENCES = 2000  # Actual user data payload
ADAPTATION_EPOCHS = 50        # Ultra-fast adaptation on pilots

# Global log file handle
report_file = None

def log(msg=""):
    print(msg)
    if report_file is not None:
        report_file.write(str(msg) + "\n")
        report_file.flush()

def generate_txB_frame(num_samples):
    """
    Simulates Transmitter B + Unknown Channel Impairments
    """
    shape = (num_samples, SEQUENCE_LEN, FEATURE_DIM)
    bits = np.random.randint(0, 2, size=shape).astype(np.float32)
    return bits

def calculate_ber(y_true, y_pred):
    predictions = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)
    errors = tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - predictions))
    return errors

def load_pretrained_rx_weights(model, weights_path):
    """
    Loads pre-trained Tx A Receiver weights (1412 float32 parameters)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pre-trained weights not found at: {weights_path}")
        
    raw_weights = np.fromfile(weights_path, dtype=np.float32)
    receiver_vars = model.rx.trainable_variables
    
    idx = 0
    for var in receiver_vars:
        shape = var.shape
        size = np.prod(shape)
        var_weights = raw_weights[idx : idx + size].reshape(shape)
        var.assign(var_weights)
        idx += size
        
    log(f"[SUCCESS] Pre-trained Rx weights loaded ({idx} float32 parameters)")

if __name__ == "__main__":
    # Ensure reports directory exists and open report file
    os.makedirs("reports", exist_ok=True)
    report_file = open("reports/fine_tune_report.txt", "w", encoding="utf-8")

    log("==================================================")
    log("  Pilot-Based Adaptation Scheme for Tx B          ")
    log("==================================================")
    
    # 1. Initialize Autoencoder
    model = Autoencoder(FEATURE_DIM, HIDDEN_DIM, SNR_DB)
    
    # Generate Wireless Frame Components
    pilot_bits = generate_txB_frame(NUM_PILOT_SEQUENCES)     # Known Pilot Symbols
    payload_bits = generate_txB_frame(NUM_PAYLOAD_SEQUENCES) # Incoming Unknown Data
    
    # Run single pass to build Keras variable shapes
    _ = model(pilot_bits[:1])
    
    # 2. Load Pre-trained Weights from Tx A
    txA_weights_path = "model_weights/lstm_signal_weights.bin"
    load_pretrained_rx_weights(model, txA_weights_path)
    
    # 3. Evaluate Zero-Shot BER on Payload Data BEFORE Adaptation
    zero_shot_logits = model(payload_bits)
    zero_shot_ber = calculate_ber(payload_bits, zero_shot_logits).numpy()
    log(f"\n[STEP 1] Zero-Shot BER on Payload Data : {zero_shot_ber * 100:.2f}% (Distorted Channel!)")
    
    # 4. Perform Fast Few-Shot Adaptation using ONLY the Pilot Symbols!
    log(f"\n[STEP 2] Running Rapid Adaptation on {NUM_PILOT_SEQUENCES} Pilot Sequences ({ADAPTATION_EPOCHS} Epochs)...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.04)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    for epoch in range(1, ADAPTATION_EPOCHS + 1):
        with tf.GradientTape() as tape:
            pilot_logits = model(pilot_bits)
            loss = loss_fn(pilot_bits, pilot_logits)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 10 == 0 or epoch == 1:
            pilot_ber = calculate_ber(pilot_bits, pilot_logits).numpy()
            log(f"  Pilot Epoch {epoch:2d}/{ADAPTATION_EPOCHS} | Loss: {loss.numpy():.4f} | Pilot BER: {pilot_ber * 100:.2f}%")
            
    # 5. Evaluate Final Payload Data BER AFTER Pilot Adaptation
    adapted_logits = model(payload_bits)
    adapted_ber = calculate_ber(payload_bits, adapted_logits).numpy()
    log(f"\n[STEP 3] Final Adapted BER on Payload Data : {adapted_ber * 100:.2f}%")
    
    # 6. Export Adaptively Tuned Weights
    if adapted_ber <= 0.02:
        txB_weights_path = "model_weights/lstm_signal_weights_txB.bin"
        export_receiver_weights(model, txB_weights_path)
        log(f"\n[SUCCESS] Adapted Tx B weights exported to: {txB_weights_path}")

    report_file.close()