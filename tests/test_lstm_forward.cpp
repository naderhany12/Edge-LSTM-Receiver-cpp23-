#include <iostream>
#include <vector>
#include <cmath>
#include "LSTM/utils.hpp"
#include "LSTM/activations.hpp"
#include "LSTM/tensor_view.hpp"
#include "LSTM/lstm_cell.hpp"

int main(){
    std::cout << "Verifying LSTM Forward Propagation" << std::endl;

    const size_t input_dim = 4;
    const size_t hidden_dim = 16;
    const size_t sequence_len = 10;

    //Load weights
    std::string weights_path = "model_weights/lstm_signal_weights.bin";
    
    std::vector<float> weights_buf = LSTM::load_binary_weights(weights_path);

    if(weights_buf.empty()){
        std::cout << "[ERROR] Failed to load weights from file: " << weights_path << std::endl;
        return -1;
    }

    size_t w_ih_size = (4 * hidden_dim) * input_dim;
    size_t w_hh_size = (4 * hidden_dim) * hidden_dim;
    size_t bias_size = (4 * hidden_dim);
    size_t dense_w_size = hidden_dim * input_dim;
    size_t dense_b_size = input_dim;


    const float* weights_ptr = weights_buf.data();
    std::span<const float> w_ih(weights_ptr, w_ih_size); weights_ptr += w_ih_size;
    std::span<const float> w_hh(weights_ptr, w_hh_size); weights_ptr += w_hh_size;
    std::span<const float> bias_lstm(weights_ptr, bias_size); weights_ptr += bias_size;
    std::span<const float> W_dense(weights_ptr, dense_w_size); weights_ptr += dense_w_size;
    std::span<const float> b_dense(weights_ptr, dense_b_size); weights_ptr += dense_b_size; 

    LSTM::LSTMCell lstm_cell(input_dim ,hidden_dim , w_ih, w_hh, bias_lstm);

    std::vector<float> h(hidden_dim, 0.0f);
    std::vector<float> c(hidden_dim, 0.0f);
    std::vector<float> dummy_noisy_signal(sequence_len * input_dim, 0.5f);

    for(size_t t = 0; t < sequence_len; t++){

        std::span<const float> x_t(dummy_noisy_signal.data() + (t * input_dim), input_dim);

        lstm_cell.step(x_t, h, c);

    }   

    std::vector<float> final_logits(input_dim, 0.0f);
    LSTM::TensorView2D dense_view(W_dense, hidden_dim, input_dim);
    LSTM::matmul_vec(dense_view, h,b_dense, final_logits);

     // Python Ground Truth Logits
    std::vector<float> python_expected = {-0.236075f, -7.46501f, 0.888805f, 1.50091f};

    //Verify output logits match Python
    for(size_t i = 0; i < input_dim; i++){
        float diff = std::abs(final_logits[i] - python_expected[i]);
        if(diff > 1e-3f){
            std::cout << "[ERROR] LSTM Forward Propagation failed! Expected: " << python_expected[i] 
                      << ", Got: " << final_logits[i] << std::endl;
            return -1;
        }
    }

        std::cout << "[PASS] Test Succeeded! Output matches Python Golden Reference." << std::endl;
        return 0;

    }