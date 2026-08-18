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


inline float dot_product(std::span<const float> a, std::span<const float> b){
    if(a.size() != b.size()){
        throw std::runtime_error("Dot Product: Vectors must have the same size!");
    }

    size_t n = a.size();

#if defined(__riscv) && defined(__riscv_vector)

    size_t v1;

    v1 =__riscv_vsetvl_e32m1(n);
    vfloat32m1_t v_acc = __riscv_vfmv_v_f_f32m1(0.0f, v1);

    size_t i = 0;
// strip-mining loop
    while(n > 0){
        v1 = __riscv_vsetvl_e32m1(n);

        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i] , v1);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i] , v1);

        v_acc = __riscv_vfmacc_vv_f32m1(v_acc, va, vb, v1);


        i += v1;
        n -= v1;

    }

    // Vector Reduction
    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_acc, v_zero,  __riscv_vsetvl_e32m1(a.size()));

    return __riscv_vfmv_f_s_f32m1_f32(v_sum);

#else

    float sum = 0.0f;
    for(size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;

#endif

}


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

    inline void matmul_vec(TensorView2D mat, std::span<const float> x, std::span<const float> bias, std::span<float> out){
        if(x.size() != mat.cols() || out.size() != mat.rows()){
            std::string err = "Matrix-Vector dimension mismatch! x.size=" + std::to_string(x.size()) +
                          ", mat.cols=" + std::to_string(mat.cols()) +
                          ", out.size=" + std::to_string(out.size()) +
                          ", mat.rows=" + std::to_string(mat.rows());
             throw std::invalid_argument(err);
            }

        for(size_t r = 0; r < mat.rows(); r++){
            float b = bias.empty() ? 0.0f : bias[r];
            out[r] = dot_product(mat.row(r), x) + b;

        }

    }

}


#endif