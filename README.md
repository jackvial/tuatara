# Setup
- Parseq weights https://huggingface.co/baudm/parseq-tiny/tree/main
- Libtorch `curl -O https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip && unzip libtorch-macos-2.0.0.zip`

# How To Run
```
cd build
cmake --build .
./tuatara ../parseq-tiny/torchscript_model.bin
```