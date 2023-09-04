cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(tuatara)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
list(APPEND CMAKE_PREFIX_PATH ${PARENT_DIR}/thirdparty/libtorch)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)

# by default build a shared library unless specified to make static library  
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Shared Libraries")

add_library(${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_sources(${PROJECT_NAME} PRIVATE tuatara.cpp utils.cpp)

target_include_directories(tuatara PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
add_subdirectory(examples)

add_subdirectory(bindings)