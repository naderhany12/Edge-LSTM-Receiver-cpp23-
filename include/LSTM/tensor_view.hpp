#ifndef TENSOR_VIEW_HPP
#define TENSOR_VIEW_HPP


#include <vector>
#include <span>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <algorithm>

//RISC-V Vectorization library
#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace LSTM {

  class TensorView2D{
    private:
        std::span<const float> data_;
        size_t rows_;
        size_t cols_;

    public:
        TensorView2D(std::span<const float> data, size_t rows, size_t cols) : 
        data_(data), rows_(rows), cols_(cols) {
            if(data.size() != rows * cols){
                throw std::invalid_argument("Data size does not match specified dimensions.");
            }
        }

        [[nodiscard]] constexpr size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr size_t cols() const noexcept { return cols_; }

        [[nodiscard]] std::span<const float> row(size_t r) const{
            if(r >= rows_){
                throw std::out_of_range("Row index out of range!");
            }
        
            return data_.subspan(r * cols_, cols_);

        }

    };

    inline void matmul_vec(TensorView2D& mat, std::span<const float> x, std::span<const float> bias, std::span<float> out){

        size_t in_dim = mat.rows();
        size_t out_dim = mat.cols();

        if(x.size() != in_dim || out.size() != out_dim){
            std::string err = "Matrix-Vector dimension mismatch! x.size=" + std::to_string(x.size()) +
                          ", mat.rows=" + std::to_string(in_dim) +
                          ", out.size=" + std::to_string(out.size()) +
                          ", mat.cols=" + std::to_string(out_dim);
             throw std::invalid_argument(err);
            }

        for (size_t j = 0; j < out_dim; ++j) {
            out[j] = bias.empty() ? 0.0f : bias[j];
            
        }

        for (size_t i = 0; i < in_dim; ++i) {
        float x_i = x[i];
        std::span<const float> row_i = mat.row(i);

#if defined(__riscv) && defined(__riscv_vector)
        size_t n = out_dim;
        size_t j = 0;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m1(n);
            vfloat32m1_t v_out = __riscv_vle32_v_f32m1(&out[j], vl);
            vfloat32m1_t v_w   = __riscv_vle32_v_f32m1(&row_i[j], vl);

            // ضرب وتجميع بـ RVV: v_out += x_i * v_w
            v_out = __riscv_vfmacc_vf_f32m1(v_out, x_i, v_w, vl);

            __riscv_vse32_v_f32m1(&out[j], v_out, vl);

            j += vl;
            n -= vl;
        }
        #else
        for (size_t j = 0; j < out_dim; ++j) {
            out[j] += x_i * row_i[j];
        }
        #endif
    }

    }

}


#endif