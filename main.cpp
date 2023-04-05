#include <iostream>
#include <memory>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// int parseq()
// {
//     std::string model_path = "/Users/jackvial/Code/CPlusPlus/torchscript_example/parseq-tiny/torchscript_model.bin";

//     // Deserialize the TorchScript module from a file
//     torch::jit::script::Module module;
//     try
//     {
//         module = torch::jit::load(model_path);
//     }
//     catch (const c10::Error &e)
//     {
//         std::cerr << "error loading the model\n";
//         return -1;
//     }

//     std::cout << "model loaded\n";

//     std::string image_path = "/Users/jackvial/Code/CPlusPlus/torchscript_example/images/art-01107.jpg";
//     cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
//     if (image.empty())
//     {
//         std::cerr << "Error reading image from file\n";
//         return -1;
//     }

//     cv::resize(image, image, cv::Size(128, 32));
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

//     torch::Tensor image_tensor = torch::from_blob(
//         image.data, {1, image.rows, image.cols, 3}, torch::kByte);

//     image_tensor = image_tensor.permute({0, 3, 1, 2}); // Rearrange dimensions to {1, 3, 32, 128}
//     image_tensor = image_tensor.to(torch::kFloat);
//     image_tensor = image_tensor.div(255.0); // Normalize pixel values (0-255 -> 0-1)

//     // Create a vector of inputs
//     std::vector<torch::jit::IValue> inputs;
//     // inputs.push_back(torch::ones({1, 3, 32, 128}));
//     inputs.push_back(image_tensor);

//     // Execute the model and turn its output into a tensor
//     at::Tensor output = module.forward(inputs).toTensor();
//     std::cout << "output: " << output << std::endl;

//     return 0;
// }

int main(int argc, const char *argv[])
{
    // if (argc != 2)
    // {
    //     std::cerr << "usage: torchscript_example <path-to-exported-script-module>\n";
    //     return -1;
    // }

    std::string model_path = "/Users/jackvial/Code/CPlusPlus/torchscript_example/weights/craft_traced_torchscript_model.pt";

    // Deserialize the TorchScript module from a file
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "model loaded\n";

    std::string image_path = "/Users/jackvial/Code/CPlusPlus/torchscript_example/images/table_english.png";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error reading image from file\n";
        return -1;
    }

    // torch.Size([1, 3, 672, 1248])

    cv::resize(image, image, cv::Size(672, 1248));
    // cv::resize(image, image, cv::Size(200, 400));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    torch::Tensor image_tensor = torch::from_blob(
        image.data, {1, image.rows, image.cols, 3}, torch::kByte);

    image_tensor = image_tensor.permute({0, 3, 1, 2}); // Rearrange dimensions to {1, 3, 32, 128}
    image_tensor = image_tensor.to(torch::kFloat);
    image_tensor = image_tensor.div(255.0); // Normalize pixel values (0-255 -> 0-1)

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor);

    // Execute the model and turn its output into a tensor
    torch::jit::IValue output_ivalue = module.forward(inputs);

    // Check if the output is a tuple
    if (output_ivalue.isTuple())
    {
        auto output_tuple = output_ivalue.toTuple();

        // Access elements in the tuple using std::get
        torch::Tensor output_tensor_1 = output_tuple->elements()[0].toTensor();
        torch::Tensor output_tensor_2 = output_tuple->elements()[1].toTensor();

        std::cout << "Done!";

        // std::cout << "Output tensor 1: " << output_tensor_1 << "\n";
        // std::cout << "Output tensor 2: " << output_tensor_2 << "\n";
    }
    else
    {
        std::cerr << "Model output is not a tuple\n";
        return -1;
    }

    return 0;
}
