cmake_minimum_required(VERSION 3.11)

project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.78 REQUIRED)

INCLUDE_DIRECTORIES("../Loader")

add_executable(NeuralNet NeuralNet3.cpp)
target_link_libraries(NeuralNet PUBLIC Boost::boost)
