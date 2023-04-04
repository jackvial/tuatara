cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchscript_example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torchscript_example main.cpp)
target_link_libraries(torchscript_example "${TORCH_LIBRARIES}")
set_property(TARGET torchscript_example PROPERTY CXX_STANDARD 14)
