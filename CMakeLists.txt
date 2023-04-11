cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(tuatara)

# # Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
set(Torch_DIR "/Users/jackvial/Code/CPlusPlus/tuatara/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find OpenCV
# Try only include only used parts of opencv with  FIND_PACKAGE( OpenCV REQUIRED core imgproc)
# See https://stackoverflow.com/questions/11196720/cmake-how-to-only-include-part-of-opencv
find_package(OpenCV 4 REQUIRED)

# Create executable
set(SOURCES main.cpp utils.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

