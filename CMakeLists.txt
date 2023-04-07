# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# project(torchscript_example)

# list(APPEND CMAKE_PREFIX_PATH "libtorch/share/cmake")

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # Find OpenCV
# find_package(OpenCV REQUIRED)

# add_executable(torchscript_example main.cpp)

# # Link libraries
# target_link_libraries(torchscript_example "${TORCH_LIBRARIES} ${OpenCV_LIBS}")

# # Include directories
# target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# set_property(TARGET torchscript_example PROPERTY CXX_STANDARD 14)




cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(opencv_torch_example)

# # Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
set(Torch_DIR "/Users/jackvial/Code/CPlusPlus/torchscript_example/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find OpenCV
# Try only include only used parts of opencv with  FIND_PACKAGE( OpenCV REQUIRED core imgproc)
# See https://stackoverflow.com/questions/11196720/cmake-how-to-only-include-part-of-opencv
find_package(OpenCV 4 REQUIRED)

# Create executable
set(SOURCES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

