#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <span>

namespace LSTM {
    // Load weights from a binary file
    inline std::vector<_Float16> load_binary_weights_fp16(const std::string& filepath) {
        std::ifstream file(filepath , std::ios::binary | std::ios::ate);

        if(!file.is_open()){
            std::cerr << "[ERROR] Could not open file: " << filepath << std::endl; 
            return{};
        }
        
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t num_floats  = file_size / sizeof(float);
        std::vector<float> fp32_buffer(num_floats);

        file.read(reinterpret_cast<char*>(fp32_buffer.data()), file_size);
        file.close();

        std::vector<_Float16> fp16_weights(num_floats);
        for(size_t i = 0; i < num_floats; ++i){
            fp16_weights[i] = static_cast<_Float16>(fp32_buffer[i]);
        }

        return fp16_weights;
    }
   

}

