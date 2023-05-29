# Tuatara: Deep Learning OCR Engine

## Design Goals
1. Easy to understand and hackable - All the main code is in tuatara.cpp and is currently about 500 LOC
2. Preference for targeting CPU over GPU and focus on CPU performance
3. Minimal dependencies and small binary - The plan is to replace both the two main dependencies OpenCV and LibTorch with [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page). I believe this will support goals 1 and 2.

# Setup and Run
## Model Weights
1. Save parseq-tiny `torchscript_model.bin` weights to the weights directory [parseq-tiny on huggingface](https://huggingface.co/baudm/parseq-tiny/tree/main) and rename to `parseq_torchscript.bin`
2. 

## Libtorch
- In root directory `curl -O https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip && unzip libtorch-macos-2.0.0.zip`

## Pybind11
- In root directory `git clone https://github.com/pybind/pybind11.git`
- `cd pybind11 && mkdir build && cd build && cmake .. && make`

## Build
```
cd build
cmake ..
make
./tuatara
```

# Run Examples
- `./run_resume_example.sh`