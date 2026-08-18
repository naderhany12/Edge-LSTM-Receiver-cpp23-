#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <span>

namespace LSTM {
    // Load weights from a binary file
    inline std::vector<float> load_binary_weights(const std::string& filepath) {
        std::ifstream file(filepath , std::ios::binary | std::ios::ate);

        if(!file.is_open()){
            std::cerr << "[ERROR] Could not open file: " << filepath << std::endl; 
            return{};
        }
        
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t num_bytes = file_size / sizeof(float);
        std::vector<float> weights(num_bytes);

        file.read(reinterpret_cast<char*>(weights.data()) , file_size);

         file.close();
        return weights;
    }
   

}

