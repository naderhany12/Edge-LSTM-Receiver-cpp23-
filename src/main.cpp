#include <iostream>
#include <vector>
#include <span>
#include <chrono>
#include <numeric>
#include "LSTM/utils.hpp"
#include "LSTM/activations.hpp"
#include "LSTM/tensor_view.hpp"
#include "LSTM/lstm_cell.hpp"

int main(){
    std::cout << "Intializing C++23 Neural Receiver Engine" << std::endl;

    //Hyperparameters
    size_t input_dim = 4;
    size_t hidden_dim = 16;
    size_t sequence_len = 10;

    //Load weights
    std::string weights_path = "../model_weights/lstm_signal_weights.bin";

    std::vector<float> weights_buf = LSTM::load_binary_weights(weights_path);

    if(weights_buf.empty()){
        std::cout << "[ERROR] Failed to load weights from file: " << weights_path << std::endl;
        return -1;
    }
    std::cout <<"Weights loaded successfully" << std::endl;

    size_t  w_ih_size = (4 * hidden_dim) * input_dim;
    size_t  w_hh_size = (4 * hidden_dim) * hidden_dim;
    size_t  bias_size = 4 * hidden_dim;

    size_t dense_w_size = hidden_dim * input_dim;
    size_t dense_b_size = input_dim;

    //Check weights size
    size_t expected_size = w_ih_size + w_hh_size + bias_size + dense_w_size + dense_b_size;
    if(weights_buf.size() != expected_size){    
         std::cerr << "[ERROR] Weights size mismatch! Expected: " << expected_size 
                  << ", Got: " << weights_buf.size() << std::endl;
                  return -1;
    }

    //Extract weights without copying
    const float* weights_ptr = weights_buf.data();
    std::span<const float> w_ih(weights_ptr, w_ih_size); weights_ptr += w_ih_size;
    std::span<const float> w_hh(weights_ptr, w_hh_size); weights_ptr += w_hh_size;
    std::span<const float> bias_lstm(weights_ptr, bias_size); weights_ptr += bias_size;
    std::span<const float> W_dense(weights_ptr, dense_w_size); weights_ptr += dense_w_size;
    std::span<const float> b_dense(weights_ptr, dense_b_size); weights_ptr += dense_b_size; 

    //Create LSTM Cell
    LSTM::LSTMCell lstm_cell(input_dim ,hidden_dim , w_ih, w_hh, bias_lstm);

    //allocate memory for hidden and cell states
    std::vector<float> h(hidden_dim, 0.0f);
    std::vector<float> c(hidden_dim, 0.0f);
     

    std::vector<float> dummy_noisy_signal(sequence_len * input_dim, 0.5f);

    std::cout<< "\n[INFO] Running Inference over " << sequence_len << " time_steps" << std::endl;

    // Measure inference latency
    auto start = std::chrono::high_resolution_clock::now();


    for(size_t t = 0; t < sequence_len; t++){

        std::span<const float> x_t(dummy_noisy_signal.data() + (t * input_dim), input_dim);

        lstm_cell.step(x_t, h, c);

    }

    std::vector<float> final_logits(input_dim, 0.0f);

    LSTM::TensorView2D dense_view(W_dense, hidden_dim, input_dim);
    LSTM::matmul_vec(dense_view, h,b_dense, final_logits);

    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "\n[PERFORMANCE] Pure Inference Latency: " << duration << " microseconds (" 
          << (float)duration / 1000.0f << " ms)" << std::endl;

        std::cout << "Raw Output Logits: ";
        for (float val : final_logits) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Decoded Bits (Threshold > 0.0): ";
        for (float val : final_logits) {
            std::cout << (val > 0.0f ? 1 : 0) << " ";
        }
        std::cout << std::endl;

        std::cout << "\n[SUCCESS] C++23 Inference Engine is fully operational!" << std::endl;
    return 0;

}