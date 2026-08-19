#pragma once
#include "activations.hpp"
#include "tensor_view.hpp"
#include <span>
#include <vector>

namespace LSTM {
    class LSTMCell{
    private :
        size_t input_dim_;
        size_t hidden_dim_;

        TensorView2D W_ih_;
        TensorView2D W_hh_;
        std::span<const _Float16> bias_;

        std::vector<_Float16> gates_buf_;


    public :
        LSTMCell(size_t input_dim, size_t hidden_dim, 
            std::span<const _Float16> W_ih, std::span<const _Float16> W_hh, std::span<const _Float16> bias) :
            input_dim_(input_dim), hidden_dim_(hidden_dim),
            W_ih_(W_ih, input_dim, 4 * hidden_dim), W_hh_(W_hh, hidden_dim, 4 * hidden_dim),
            bias_(bias), gates_buf_(4 * hidden_dim)
        {}

        
            void step(std::span<const _Float16> x_t, std::span<_Float16> h, std::span<_Float16> c){
                // input gate
                matmul_vec(W_ih_, x_t, bias_, gates_buf_ , false);
                
                // hidden gate
                matmul_vec(W_hh_, h, {}, gates_buf_, true);


                //Apply activation function
                for(size_t i = 0; i < hidden_dim_; i++){
                
                _Float16 g_in  = gates_buf_[i];
                _Float16 g_f   = gates_buf_[i + hidden_dim_];
                _Float16 g_c   = gates_buf_[i + 2 * hidden_dim_];
                _Float16 g_out = gates_buf_[i + 3 * hidden_dim_];

                _Float16 in_gate     = sigmoid(g_in);
                _Float16 forget_gate = sigmoid(g_f);
                _Float16 cell_gate   = tanh_act(g_c);
                _Float16 out_gate    = sigmoid(g_out);



                    c[i] = (forget_gate * c[i]) + (in_gate * cell_gate);
                    h[i] = out_gate * tanh_act(c[i]);       
                }

        }


};

}