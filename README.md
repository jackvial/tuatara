# Tuatara: Deep Learning OCR Engine

## Design Goals
1. Easy to understand and hackable - All the main code is in tuatara.cpp and is currently about 500 LOC
2. Preference for targeting CPU over GPU and focus on CPU performance
3. Minimal dependencies and small binary - The plan is to replace both the two main dependencies OpenCV and LibTorch with [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page). I believe this will support goals 1 and 2.

## Setup
Run `./setup.sh` or do following steps
1. Download model weights `git clone https://huggingface.co/jackvial/tuatara-ocr-craft-and-parseq weights`
2. Download LibTorch `curl -O https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip && unzip libtorch-macos-2.0.0.zip`
3. Install opencv `brew install opencv`
4. Download and build pybind11 `git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make`

## Build
```
mkdir -p build
cd build
cmake ..
```

## Run Example with Python Bindings
- `cd bindings && python3 run_ocr.py`

## Debug C++ in VSCode
- Install the CodeLLDB VSCode extension
- Set a breakpoint in examples/resume.cpp or tuatara.cpp
- Run the "Debug Resume Example" under the VSCode "Run and Debug" tab