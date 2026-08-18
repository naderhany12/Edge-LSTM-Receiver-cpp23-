
cmake_minimum_required(VERSION 3.23)
project(LSTM_Edge_Inference LANGUAGES CXX)



set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "-O3")
endif()

add_executable(LSTM_Edge_Inference src/main.cpp)

enable_testing()
add_subdirectory(tests)