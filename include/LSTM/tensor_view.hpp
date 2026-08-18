# pragma once
# include <span>
# include <cstddef>

namespace LSTM {

    class TensorView2D{
    private:
        std::span<const float> data_;
        size_t rows_;
        size_t cols_;

    public:
        TensorView2D(std::span<const float> data, size_t rows, size_t cols) : 
        data_(data), rows_(rows), cols_(cols) {}

        inline float operator()(size_t row, size_t c) const{
            return data_[row * cols_ + c];
        }

        inline size_t rows() const{ return rows_; }
        inline size_t cols() const{ return cols_; }
    };


inline void matmul_vec(
    const TensorView2D& W,
    std::span<const float> x,
    std::span<const float> bias,
    std::span<float> out
){

    for(size_t c = 0; c < W.cols(); c++){
         float sum = (bias.empty()) ? 0.0f : bias[c];
        for(size_t r = 0; r < W.rows(); r++)
                 sum += W(r,c) * x[r];

        out[c] = sum;
    }

}

}