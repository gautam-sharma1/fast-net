cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FastNet)


add_executable(main src/main.cpp src/Tensor.cpp src/Layer.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
