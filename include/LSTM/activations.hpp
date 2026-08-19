#pragma once
    
#include <cmath>
#include <span>

namespace LSTM {
    // Sigmoid activation function
    inline _Float16  sigmoid(_Float16  x) noexcept {
        float x_f32 =  static_cast<float>(x); 
        return static_cast<_Float16>(1.0f / (1.0f + std::exp(-x_f32)));
    }

    // Tanh activation function
    inline _Float16  tanh_act(_Float16  x ) noexcept {
        float x_f32 = static_cast<float>(x);
        return static_cast<_Float16>(std::tanh(x_f32));
    }

    inline void apply_sigmoid(std::span<_Float16> data){
        for(auto& x: data){
            x = sigmoid(x);
        }       

    }

    inline void apply_tanh_act(std::span<_Float16> data){
        for(auto& x: data){
            x = tanh_act(x);
        }
    }

}