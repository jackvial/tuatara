#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <utility>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace torch::indexing;

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

void print_tensor_dims(std::string label, torch::Tensor t)
{
    int64_t num_dims = t.dim();
    // Print the dimensions of the tensor
    std::cout << label << " (";
    for (int64_t i = 0; i < num_dims; ++i)
    {
        std::cout << t.size(i);
        if (i < num_dims - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}

void display_2d_tensor_heatmap(torch::Tensor t)
{
    // Normalize the tensor to the range [0, 1]
    torch::Tensor tensor_normalized = (t - t.min()) / (t.max() - t.min());

    // Convert the normalized tensor to an OpenCV Mat
    cv::Mat mat(tensor_normalized.size(0), tensor_normalized.size(1), CV_32F, tensor_normalized.data_ptr<float>());

    // Convert the normalized Mat to a heatmap
    cv::Mat heatmap;
    mat.convertTo(heatmap, CV_8UC1, 255);
    cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);

    // Display the heatmap
    cv::imshow("Heatmap", heatmap);
    cv::waitKey(0);
}

std::pair<std::vector<cv::RotatedRect>, cv::Mat> get_detected_boxes(
    torch::Tensor textmap, torch::Tensor linkmap, float text_threshold,
    float link_threshold, float low_text, bool estimate_num_chars = false)
{
    // Convert the normalized tensor to an OpenCV Mat
    cv::Mat textmap_cv(textmap.size(0), textmap.size(1), CV_32F, textmap.data_ptr<float>());
    // cv::Mat textmap_cv = textmap.to(torch::kF32).mul(255).clamp(0, 255).to(torch::kU8).squeeze().detach().numpy().clone();

    cv::Mat linkmap_cv(linkmap.size(0), linkmap.size(1), CV_32F, linkmap.data_ptr<float>());
    // cv::Mat linkmap_cv = linkmap.to(torch::kF32).mul(255).clamp(0, 255).to(torch::kU8).squeeze().detach().numpy().clone();

    int img_h = textmap_cv.rows;
    int img_w = textmap_cv.cols;

    cv::Mat text_score, link_score;
    cv::threshold(textmap_cv, text_score, low_text, 1, 0);
    cv::threshold(linkmap_cv, link_score, link_threshold, 1, 0);

    cv::Mat text_score_comb = cv::min(text_score + link_score, 1);
    text_score_comb.convertTo(text_score_comb, CV_8U);

    cv::Mat labels, stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(text_score_comb, labels, stats, centroids);

    std::vector<cv::RotatedRect> det;
    std::vector<int> mapper;
    for (int k = 1; k < nLabels; ++k)
    {
        int size = stats.at<int>(k, cv::CC_STAT_AREA);
        if (size < 10)
            continue;

        cv::Mat mask = (labels == k);
        double minVal, maxVal;
        cv::minMaxLoc(textmap_cv, &minVal, &maxVal, nullptr, nullptr, mask);

        if (maxVal < text_threshold)
            continue;

        cv::Mat segmap = cv::Mat::zeros(textmap_cv.size(), CV_8UC1);
        segmap.setTo(255, labels == k);

        if (estimate_num_chars)
        {
            cv::Mat character_locs;
            cv::threshold(((textmap_cv - linkmap_cv).mul(segmap) / 255.0), character_locs, text_threshold, 1, 0);
            cv::Mat labels_characters;
            int n_chars = cv::connectedComponents(character_locs, labels_characters);
            mapper.push_back(n_chars);
        }
        else
        {
            mapper.push_back(k);
        }

        segmap.setTo(0, (link_score == 1) & (text_score == 0));

        int x = stats.at<int>(k, cv::CC_STAT_LEFT);
        int y = stats.at<int>(k, cv::CC_STAT_TOP);
        int w = stats.at<int>(k, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        int niter = static_cast<int>(std::sqrt(size * std::min(w, h) / (w * h) * 2));

        int sx = std::max(0, x - niter);
        int sy = std::max(0, y - niter);
        int ex = std::min(img_w, x + w + niter + 1);
        int ey = std::min(img_h, y + h + niter + 1);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
        cv::dilate(segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), kernel);

        std::vector<cv::Point> np_contours;
        cv::findNonZero(segmap, np_contours);
        cv::RotatedRect rectangle = cv::minAreaRect(np_contours);
        cv::Point2f box[4];
        rectangle.points(box);

        float box_w = cv::norm(box[0] - box[1]);
        float box_h = cv::norm(box[1] - box[2]);
        float box_ratio = std::max(box_w, box_h) / (std::min(box_w, box_h) + 1e-5);

        if (std::abs(1 - box_ratio) <= 0.1)
        {
            int l = std::min_element(np_contours.begin(), np_contours.end(),
                                     [](const cv::Point &a, const cv::Point &b)
                                     { return a.x < b.x; })
                        ->x;
            int r = std::max_element(np_contours.begin(), np_contours.end(),
                                     [](const cv::Point &a, const cv::Point &b)
                                     { return a.x < b.x; })
                        ->x;
            int t = std::min_element(np_contours.begin(), np_contours.end(),
                                     [](const cv::Point &a, const cv::Point &b)
                                     { return a.y < b.y; })
                        ->y;
            int b = std::max_element(np_contours.begin(), np_contours.end(),
                                     [](const cv::Point &a, const cv::Point &b)
                                     { return a.y < b.y; })
                        ->y;

            cv::Point2f new_box[4] = {cv::Point2f(l, t), cv::Point2f(r, t), cv::Point2f(r, b), cv::Point2f(l, b)};
            std::copy(new_box, new_box + 4, box);
        }

        int startidx = std::distance(box, std::min_element(box, box + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                                                           { return a.x + a.y < b.x + b.y; }));
        std::rotate(box, box + startidx, box + 4);

        det.emplace_back(rectangle);
    }

    return std::make_pair(det, labels);
}

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
    cv::resize(image, image, cv::Size(312 * 4, 168 * 4));
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

        print_tensor_dims(" output_tensor_1 ", output_tensor_1);

        // Get the size of the batch dimension
        int64_t batch_size = output_tensor_1.size(0);

        // Iterate through the batch dimension
        for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            // Access the tensor corresponding to the current batch
            torch::Tensor current_batch = output_tensor_1[batch_index];
            print_tensor_dims(" current_batch ", current_batch);

            // Separate the tensor into two tensors of dimensions 624 x 336
            torch::Tensor score_text = current_batch.slice(2, 0, 1).squeeze(2);
            torch::Tensor score_link = current_batch.slice(2, 1, 2).squeeze(2);

            print_tensor_dims(" score_text ", score_text);
            print_tensor_dims(" score_link ", score_link);

            // display_2d_tensor_heatmap(score_text);
            // display_2d_tensor_heatmap(score_link);

            // Set parameters for the function
            float text_threshold = 0.7;
            float link_threshold = 0.4;
            float low_text = 0.1;
            bool estimate_num_chars = false;
            auto result = get_detected_boxes(score_text, score_link, text_threshold, link_threshold, low_text, estimate_num_chars);

            // Extract results
            auto det = result.first;
            auto labels = result.second;

            // Print results
            std::cout << "Detected boxes: " << det.size() << std::endl;
            for (const auto &box : det)
            {
                cv::Point2f corners[4];
                box.points(corners);
                std::cout << "Box: ";
                for (int i = 0; i < 4; ++i)
                {
                    std::cout << "(" << corners[i].x << ", " << corners[i].y << ") ";
                }
                std::cout << std::endl;
            }

            // Draw the detected boxes on the image
            for (const auto &box : det)
            {
                cv::Point2f corners[4];
                box.points(corners);
                std::vector<cv::Point> corners_vec(corners, corners + 4);
                cv::polylines(image, corners_vec, true, cv::Scalar(0, 255, 0), 2);
            }

            // Display the image with the drawn boxes
            cv::namedWindow("Detected Boxes", cv::WINDOW_NORMAL);
            cv::imshow("Detected Boxes", image);
            cv::waitKey(0);
        }

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
