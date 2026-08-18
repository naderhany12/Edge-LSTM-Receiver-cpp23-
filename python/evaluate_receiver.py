import os
import numpy as np
import tensorflow as tf
from models import Receiver

SEQUENCE_LEN = 10
FEATURE_DIM = 4
HIDDEN_DIM = 16
WEIGHTS_PATH = "../model_weights/lstm_signal_weights.bin"   

def load_weights(filepath):
    if not os.path.exists(filepath):
        raise   FileNotFoundError(f"File not found: {filepath}")    
    # Load weights from binary file
    
    raw_data = np.fromfile(filepath, dtype=np.float32)

    w_ih_size = FEATURE_DIM * (HIDDEN_DIM * 4)
    w_hh_size = HIDDEN_DIM * (HIDDEN_DIM * 4)
    bias_lstm_size = HIDDEN_DIM * 4
    w_dense_size = HIDDEN_DIM * FEATURE_DIM
    b_dense_size = FEATURE_DIM

    idx = 0

    w_ih = raw_data[idx : idx + w_ih_size].reshape(FEATURE_DIM, HIDDEN_DIM * 4)
    idx += w_ih_size

    w_hh = raw_data[idx : idx + w_hh_size].reshape(HIDDEN_DIM, HIDDEN_DIM * 4)
    idx += w_hh_size

    bias_lstm = raw_data[idx : idx + bias_lstm_size]
    idx += bias_lstm_size

    w_dense = raw_data[idx : idx + w_dense_size].reshape(HIDDEN_DIM, FEATURE_DIM)
    idx += w_dense_size

    b_dense = raw_data[idx : idx + b_dense_size]

    return (w_ih, w_hh, bias_lstm) ,  (w_dense, b_dense)


def run_evaluation():
    print(f"--- Loading weights from: {WEIGHTS_PATH} ---")

    lstm_weights, dense_weights = load_weights(WEIGHTS_PATH)

    receiver = Receiver(HIDDEN_DIM, FEATURE_DIM)

    dummy_input = tf.zeros((1, SEQUENCE_LEN, FEATURE_DIM), dtype=tf.float32)
    _ = receiver(dummy_input)
    
    receiver.lstm.set_weights(lstm_weights)
    receiver.dense.set_weights(dense_weights)
    print(f"--- Weights loaded successfully ---")

    test_input = np.full((1, SEQUENCE_LEN, FEATURE_DIM), 0.5, dtype=np.float32)
    output_logits = receiver(test_input).numpy()


    last_step_logits = output_logits[0, -1, :]
    print("\n=== Python Receiver Output (Last Step) ===")
    print("Raw Output Logits :", np.round(last_step_logits, 5))
    print("Decoded Bits (>0) :", (last_step_logits > 0).astype(int))

if __name__ == "__main__":
    run_evaluation();

    