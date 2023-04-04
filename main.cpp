#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: torchscript_example <path-to-exported-script-module>\n";
        return -1;
    }

    // Deserialize the TorchScript module from a file
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "model loaded\n";

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 32, 128}));

    // Execute the model and turn its output into a tensor
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "output: " << output << std::endl;

    return 0;
}
