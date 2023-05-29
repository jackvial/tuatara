#!/bin/bash

# Download the weights if weights directory does not exist
if [ ! -d "weights" ]; then
    mkdir weights
    git clone https://huggingface.co/jackvial/tuatara-ocr-craft-and-parseq weights
    echo "Downloaded model weights to the weights directory."
else
    echo "The weights directory already exists. Skipping download."
fi

# Download libtorch
if [ ! -d "libtorch" ]; then
    curl -O https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip && unzip libtorch-macos-2.0.0.zip
    echo "Downloaded libtorch to the libtorch directory."
    rm libtorch-macos-2.0.0.zip
else
    echo "The libtorch directory already exists. Skipping download."
fi

# Install opencv on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing dependencies for macOS..."
    brew install opencv
fi

# Install opencv on linux (only tested on debian/ubuntu)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing dependencies for linux..."
    sudo apt-get install libopencv-dev
fi

# Download pybind11
if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make
else
    echo "The pybind11 directory already exists. Skipping download."
fi