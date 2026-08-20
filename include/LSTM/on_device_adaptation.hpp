#pragma once
#include <span>
#include <cmath>
#include <vector>
#include "LSTM/activations.hpp"

namespace LSTM {


    inline void on_device_adaptation(
        std::span<const float> h,
        std::span<const float> y_true,
        std::span<float> W_dense,
        std::span<float> b_dense,
        size_t hidden_dim,
        size_t output_dim,
        float learning_rate)
    {
        // forward pass 
        std::vector<float> logits(output_dim , 0.0f);
        std::vector<float> y_pred(output_dim , 0.0f);

        for (size_t i = 0; i < output_dim ; i++){
            float sum = b_dense[i];
            for (size_t j = 0 ; j < hidden_dim ; j++)
                sum += h[j] * W_dense[i + j * output_dim];

            logits[i] = sum;
            y_pred[i] = LSTM::sigmoid(sum);
        }

        // Analytical gradient computation
        for (size_t i = 0; i < output_dim; i++){

            float error = y_pred[i] - y_true[i];

            // bias = bias - learning_rate * error
            b_dense[i] -= learning_rate * error;

            for (size_t j = 0; j < hidden_dim; j++){
                float grad_w = h[j] * error;

                // W = W - learning_rate * error * h
                W_dense[i + j * output_dim] -= learning_rate * grad_w;

            }
        }
    }
}