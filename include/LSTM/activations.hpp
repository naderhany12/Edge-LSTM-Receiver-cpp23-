#pragma once
    
#include <cmath>
#include <span>

namespace LSTM {
    // Sigmoid activation function
    inline float sigmoid(float x){
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Tanh activation function
    inline float tanh_act(float x ){
        return std::tanh(x);
    }

    inline void apply_sigmoid(std::span<float> data){
        for(auto& x: data){
            x = sigmoid(x);
        }       

    }

    inline void apply_tanh_act(std::span<float> data){
        for(auto& x: data){
            x = tanh_act(x);
        }
    }

}