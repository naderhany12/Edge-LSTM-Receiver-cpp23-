import os
import numpy as np


def export_receiver_weights(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract Receiver weights (LSTM + Dense)
    lstm_weights = model.rx.lstm.get_weights()
    w_ih = lstm_weights[0]
    w_hh = lstm_weights[1]
    bias_lstm = lstm_weights[2]

    dense_weights = model.rx.dense.get_weights()
    w_dense = dense_weights[0]
    b_dense = dense_weights[1]

    # Save as raw float32 binary file
    with open(output_path, "wb") as f:
        f.write(w_ih.astype(np.float32).tobytes())
        f.write(w_hh.astype(np.float32).tobytes())
        f.write(bias_lstm.astype(np.float32).tobytes())
        f.write(w_dense.astype(np.float32).tobytes())
        f.write(b_dense.astype(np.float32).tobytes())

    print(f"[SUCCESS] Receiver weights exported successfully to: {output_path}")