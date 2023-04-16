cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(tuatara)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_library(tuatara STATIC
    tuatara.cpp
    utils.cpp
)

target_include_directories(tuatara PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
add_subdirectory(examples)

add_subdirectory(bindings)