#include <iostream>
#include <vector>
#include <span>
#include <chrono>
#include <random>
#include <cmath>
#include "LSTM/utils.hpp"
#include "LSTM/activations.hpp"
#include "LSTM/tensor_view.hpp"
#include "LSTM/lstm_cell.hpp"
#include "LSTM/on_device_adaptation.hpp"

constexpr size_t INPUT_DIM = 4;
constexpr size_t HIDDEN_DIM = 16;
constexpr size_t SEQUENCE_LEN = 10;
constexpr size_t NUM_PILOT_SEQUENCES = 100;    // 400 Pilot Bits
constexpr size_t NUM_PAYLOAD_SEQUENCES = 1000;  // 4000 Payload Bits
constexpr size_t ADAPTATION_EPOCHS = 20;
constexpr float LEARNING_RATE = 0.02f;
constexpr float sigma = 0.35f;


void generate_txB_frame(size_t num_sequences, std::vector<float>& noisy_signals, std::vector<float>& target_bits) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> bit_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, sigma);

    noisy_signals.resize(num_sequences * SEQUENCE_LEN * INPUT_DIM);
    target_bits.resize(num_sequences * INPUT_DIM);

    for (size_t seq = 0; seq < num_sequences; ++seq) {
        for (size_t d = 0; d < INPUT_DIM; ++d) {
            float bit = (bit_dist(gen) > 0.5f) ? 1.0f : 0.0f;
            target_bits[seq * INPUT_DIM + d] = bit;

            for (size_t t = 0; t < SEQUENCE_LEN; ++t) {
                size_t idx = seq * (SEQUENCE_LEN * INPUT_DIM) + t * INPUT_DIM + d;
                float TxB_distortion = (bit > 0.5f) ? 0.75f : -0.75f;
                noisy_signals[idx] = TxB_distortion + noise_dist(gen);
            }
        }
    }
}

float calculate_dataset_ber(
    LSTM::LSTMCell& lstm_cell,
    std::span<const float> W_dense,
    std::span<const float> b_dense,
    const std::vector<float>& signals,
    const std::vector<float>& targets,
    size_t num_sequences) 
{
    size_t total_bit_errors = 0;
    size_t total_bits = num_sequences * INPUT_DIM;
    LSTM::TensorView2D dense_view(W_dense, HIDDEN_DIM, INPUT_DIM);

    for (size_t seq = 0; seq < num_sequences; ++seq) {
        std::vector<float> h(HIDDEN_DIM, 0.0f);
        std::vector<float> c(HIDDEN_DIM, 0.0f);

        const float* seq_signal_ptr = signals.data() + seq * (SEQUENCE_LEN * INPUT_DIM);
        const float* seq_target_ptr = targets.data() + seq * INPUT_DIM;

        for (size_t t = 0; t < SEQUENCE_LEN; ++t) {
            std::span<const float> x_t(seq_signal_ptr + (t * INPUT_DIM), INPUT_DIM);
            lstm_cell.step(x_t, h, c);
        }

        std::vector<float> logits(INPUT_DIM, 0.0f);
        LSTM::matmul_vec(dense_view, h, b_dense, logits);

        for (size_t d = 0; d < INPUT_DIM; ++d) {
            float decoded = (logits[d] > 0.0f) ? 1.0f : 0.0f;
            if (decoded != seq_target_ptr[d]) total_bit_errors++;
        }
    }
    return (static_cast<float>(total_bit_errors) / total_bits) * 100.0f;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  C++23 Real-Time Wireless Pilot Adaptation Suite (Tx B)  " << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::string weights_path = "model_weights/lstm_signal_weights.bin";
    std::vector<float> weights_buf = LSTM::load_binary_weights(weights_path);

    if (weights_buf.empty()) {
        std::cerr << "[ERROR] Could not load base weights!" << std::endl;
        return -1;
    }

    size_t w_ih_size = (4 * HIDDEN_DIM) * INPUT_DIM;
    size_t w_hh_size = (4 * HIDDEN_DIM) * HIDDEN_DIM;
    size_t bias_size = 4 * HIDDEN_DIM;
    size_t dense_w_size = HIDDEN_DIM * INPUT_DIM;
    size_t dense_b_size = INPUT_DIM;

    const float* weights_ptr = weights_buf.data();
    std::span<const float> w_ih(weights_ptr, w_ih_size); weights_ptr += w_ih_size;
    std::span<const float> w_hh(weights_ptr, w_hh_size); weights_ptr += w_hh_size;
    std::span<const float> bias_lstm(weights_ptr, bias_size); weights_ptr += bias_size;

    float* mutable_ptr = weights_buf.data() + w_ih_size + w_hh_size + bias_size;
    std::span<float> W_dense(mutable_ptr, dense_w_size); mutable_ptr += dense_w_size;
    std::span<float> b_dense(mutable_ptr, dense_b_size);

    LSTM::LSTMCell lstm_cell(INPUT_DIM, HIDDEN_DIM, w_ih, w_hh, bias_lstm);

    std::vector<float> pilot_signals, pilot_targets;
    std::vector<float> payload_signals, payload_targets;
    generate_txB_frame(NUM_PILOT_SEQUENCES, pilot_signals, pilot_targets);
    generate_txB_frame(NUM_PAYLOAD_SEQUENCES, payload_signals, payload_targets);

    float zero_shot_ber = calculate_dataset_ber(lstm_cell, W_dense, b_dense, payload_signals, payload_targets, NUM_PAYLOAD_SEQUENCES);
    std::cout << "\n[STEP 1] Zero-Shot Payload BER: " << zero_shot_ber << "%" << std::endl;

    std::cout << "\n[STEP 2] Running Rapid On-Device Adaptation at σ = " << sigma << "..." << std::endl;


    LSTM::AdamState adam_opt(HIDDEN_DIM, INPUT_DIM);

    auto adapt_start = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 1; epoch <= ADAPTATION_EPOCHS; ++epoch) {
        for (size_t seq = 0; seq < NUM_PILOT_SEQUENCES; ++seq) {
            std::vector<float> h(HIDDEN_DIM, 0.0f);
            std::vector<float> c(HIDDEN_DIM, 0.0f);

            const float* seq_signal_ptr = pilot_signals.data() + seq * (SEQUENCE_LEN * INPUT_DIM);
            const float* seq_target_ptr = pilot_targets.data() + seq * INPUT_DIM;

            for (size_t t = 0; t < SEQUENCE_LEN; ++t) {
                std::span<const float> x_t(seq_signal_ptr + (t * INPUT_DIM), INPUT_DIM);
                lstm_cell.step(x_t, h, c);
            }

            std::span<const float> y_true(seq_target_ptr, INPUT_DIM);
            LSTM::on_device_adaptation(h, y_true, W_dense, b_dense, adam_opt, HIDDEN_DIM, INPUT_DIM, LEARNING_RATE);
        }

        if (epoch % 10 == 0 || epoch == 1) {
            float pilot_ber = calculate_dataset_ber(lstm_cell, W_dense, b_dense, pilot_signals, pilot_targets, NUM_PILOT_SEQUENCES);
            std::cout << "  C++ Pilot Epoch " << epoch << "/" << ADAPTATION_EPOCHS << " | Pilot Dataset BER: " << pilot_ber << "%" << std::endl;
        }
    }

    auto adapt_end = std::chrono::high_resolution_clock::now();
    auto adapt_duration = std::chrono::duration_cast<std::chrono::microseconds>(adapt_end - adapt_start).count();

    float final_payload_ber = calculate_dataset_ber(lstm_cell, W_dense, b_dense, payload_signals, payload_targets, NUM_PAYLOAD_SEQUENCES);

    std::cout << "\n[STEP 3] Final Adapted Payload BER (4000 bits): " << final_payload_ber << "%" << std::endl;
    std::cout << "[PERFORMANCE] Total Adaptation Latency: " << adapt_duration << " us (" << static_cast<float>(adapt_duration) / 1000.0f << " ms)" << std::endl;

    return 0;
}