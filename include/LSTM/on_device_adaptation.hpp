#pragma once
#include <span>
#include <cmath>
#include <vector>
#include <array>
#include "LSTM/activations.hpp"

namespace LSTM {

    template<size_t hidden_dim = 16, size_t output_dim = 4>
    struct AdamState{
        static constexpr size_t W_size = hidden_dim * output_dim;

        std::array<float, W_size> m_W{}; // First moment vector for W_dense (size 64)
        std::array<float, W_size> v_W{}; // Second moment vector for W_dense (size 64)
        std::array<float, output_dim> m_b{}; // First moment vector for b_dense (size 4)
        std::array<float, output_dim> v_b{}; // Second moment vector for b_dense (size 4)
        size_t t{0};            // Time step / iteration counter
        float beta1_pow{1.0f};  // Adam beta1 power
        float beta2_pow{1.0f};  // Adam beta2 power

         AdamState() = default;
        
    

    // Reset Adam state
    void reset() {
            m_W.fill(0.0f);
            v_W.fill(0.0f);
            m_b.fill(0.0f);
            v_b.fill(0.0f);
            t = 0;
            beta1_pow = 1.0f;
            beta2_pow = 1.0f;
        }
    };


    template<size_t hidden_dim = 16, size_t output_dim = 4>
    inline void on_device_adaptation(
        std::span<const float> h,
        std::span<const float> y_true,
        std::span<float> W_dense,
        std::span<float> b_dense,
        AdamState<hidden_dim, output_dim>& adam,
        float learning_rate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f)

    {
        adam.t++;

        adam.beta1_pow *= beta1;
        adam.beta2_pow *= beta2;

        // forward pass 
        std::array<float, output_dim> y_pred;

        for (size_t i = 0; i < output_dim ; i++){
            float sum = b_dense[i];
            for (size_t j = 0 ; j < hidden_dim ; j++)
                sum += h[j] * W_dense[i + j * output_dim];

            y_pred[i] = LSTM::sigmoid(sum);
        }

        // Pre-compute Adam bias correction terms
        const float bias_correction1 = 1.0f - adam.beta1_pow;
        const float bias_correction2 = 1.0f - adam.beta2_pow;

        // Analytical gradient computation
        for (size_t i = 0; i < output_dim; i++){

            float error = y_pred[i] - y_true[i];

            // update Bias with Adam
            float g_b = error;
            adam.m_b[i] = beta1 * adam.m_b[i] + (1.0f - beta1) * g_b;
            adam.v_b[i] = beta2 * adam.v_b[i] + (1.0f - beta2) * (g_b * g_b);

            float m_hat_b = adam.m_b[i] / bias_correction1;
            float v_hat_b = adam.v_b[i] / bias_correction2;

            b_dense[i] -= learning_rate * (m_hat_b / (std::sqrt(v_hat_b) + epsilon));

            // update Weights with Adam
            for (size_t j = 0; j < hidden_dim; j++){
                size_t idx = i + j * output_dim;
                float g_w = h[j] * error;

                // W = W - learning_rate * error * h
                adam.m_W[idx] = beta1 * adam.m_W[idx] + (1.0f - beta1) * g_w;
                adam.v_W[idx] = beta2 * adam.v_W[idx] + (1.0f - beta2) * (g_w * g_w);

                float m_hat_w = adam.m_W[idx] / bias_correction1;
                float v_hat_w = adam.v_W[idx] / bias_correction2;

                W_dense[idx] -= learning_rate * (m_hat_w / (std::sqrt(v_hat_w) + epsilon));
            }
        }
    }
}