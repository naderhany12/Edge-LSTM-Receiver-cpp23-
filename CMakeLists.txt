
cmake_minimum_required(VERSION 3.23)
project(LSTM_Edge_Inference LANGUAGES CXX)



set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

add_executable(LSTM_Edge_Inference src/main.cpp)

enable_testing()
add_subdirectory(tests)