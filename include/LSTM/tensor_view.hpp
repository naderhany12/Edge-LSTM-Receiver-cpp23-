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
        std::span<const _Float16> data_;
        size_t rows_;
        size_t cols_;

    public:
        TensorView2D(std::span<const _Float16> data, size_t rows, size_t cols) : 
        data_(data), rows_(rows), cols_(cols) {
            if(data.size() != rows * cols){
                throw std::invalid_argument("Data size does not match specified dimensions.");
            }
        }

        [[nodiscard]] constexpr size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr size_t cols() const noexcept { return cols_; }

        [[nodiscard]] std::span<const _Float16> row(size_t r) const{
            if(r >= rows_){
                throw std::out_of_range("Row index out of range!");
            }
        
            return data_.subspan(r * cols_, cols_);

        }

    };

    inline void matmul_vec(const TensorView2D& mat, 
                       std::span<const _Float16> x, 
                       std::span<const _Float16> bias, 
                       std::span<_Float16> out,
                        bool accumulate = false) {
                size_t in_dim = mat.rows();
                size_t out_dim = mat.cols();

        #if defined(__riscv) && defined(__riscv_vector)
            size_t j = 0;
            size_t n = out_dim;

            while (n > 0) {
                size_t vl = __riscv_vsetvl_e16m1(n);

                vfloat16m1_t v_acc;
                if(accumulate){
                    v_acc = __riscv_vle16_v_f16m1(&out[j], vl);
                }
                else if (!bias.empty()) {
                    v_acc = __riscv_vle16_v_f16m1(&bias[j], vl);
                } else {
                    v_acc = __riscv_vfmv_v_f_f16m1(static_cast<_Float16>(0.0f), vl);
                }

                const _Float16* mat_ptr = mat.row(0).data();

                for (size_t i = 0; i < in_dim; ++i) {
                    _Float16 x_i = x[i];
                    const _Float16* row_ptr = mat_ptr + (i * out_dim);
                    vfloat16m1_t v_w = __riscv_vle16_v_f16m1(&row_ptr[j], vl);
                    v_acc = __riscv_vfmacc_vf_f16m1(v_acc, x_i, v_w, vl);
                }

                __riscv_vse16_v_f16m1(&out[j], v_acc, vl);

                j += vl;
                n -= vl;
            }
        #else
            if (!accumulate) {
                for (size_t j = 0; j < out_dim; ++j) {
                    out[j] = bias.empty() ? static_cast<_Float16>(0.0f) : bias[j];
                }
            }
            const _Float16* mat_ptr = mat.row(0).data();
            for (size_t i = 0; i < in_dim; ++i) {
                _Float16  x_i = x[i];
                const _Float16 * row_ptr = mat_ptr + (i * out_dim);
                for (size_t j = 0; j < out_dim; ++j) {
                    out[j] += x_i * row_ptr[j];
                }
            }
        #endif
        }

}


#endif