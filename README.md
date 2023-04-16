# Setup
## Weights
- Save parseq weights to weights directory https://huggingface.co/baudm/parseq-tiny/tree/main

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