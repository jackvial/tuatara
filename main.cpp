#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <utility>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"

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

std::pair<std::vector<cv::RotatedRect>, cv::Mat> get_detected_boxes(
    torch::Tensor textmap, torch::Tensor linkmap, float text_threshold,
    float link_threshold, float low_text, bool estimate_num_chars = false)
{
    torch::Tensor textmap_normalized = (textmap - textmap.min()) / (textmap.max() - textmap.min());
    torch::Tensor linkmap_normalized = (linkmap - linkmap.min()) / (linkmap.max() - linkmap.min());

    // Convert the normalized tensor to an OpenCV Mat
    cv::Mat textmap_cv(textmap_normalized.size(0), textmap_normalized.size(1), CV_32F, textmap_normalized.data_ptr<float>());
    cv::Mat linkmap_cv(linkmap_normalized.size(0), linkmap_normalized.size(1), CV_32F, linkmap_normalized.data_ptr<float>());

    int img_h = textmap_cv.rows;
    int img_w = textmap_cv.cols;

    cv::Mat text_score, link_score;
    cv::threshold(textmap_cv, text_score, low_text, 1, 0);
    cv::threshold(linkmap_cv, link_score, link_threshold, 1, 0);

    cv::Mat text_score_comb = cv::min(cv::max(text_score + link_score, 0.0), 1.0);
    text_score_comb.convertTo(text_score_comb, CV_8U);

    cv::Mat labels, stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(text_score_comb, labels, stats, centroids, 4);

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

std::tuple<cv::Mat, float, cv::Size> resize_aspect_ratio(
    const cv::Mat &img, int square_size, int interpolation, float mag_ratio = 1)
{
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();

    float target_size = mag_ratio * std::max(height, width);

    if (target_size > square_size)
    {
        target_size = square_size;
    }

    float ratio = target_size / std::max(height, width);

    int target_h = static_cast<int>(height * ratio);
    int target_w = static_cast<int>(width * ratio);

    cv::Mat proc;
    cv::resize(img, proc, cv::Size(target_w, target_h), 0, 0, interpolation);

    int target_h32 = target_h % 32 != 0 ? target_h + (32 - target_h % 32) : target_h;
    int target_w32 = target_w % 32 != 0 ? target_w + (32 - target_w % 32) : target_w;

    cv::Mat resized = cv::Mat::zeros(target_h32, target_w32, img.type());
    proc.copyTo(resized(cv::Rect(0, 0, target_w, target_h)));

    cv::Size size_heatmap(target_w / 2, target_h / 2);

    return std::make_tuple(resized, ratio, size_heatmap);
}

std::vector<cv::RotatedRect> adjust_result_coordinates(const std::vector<cv::RotatedRect> &polys, float ratio_w, float ratio_h, float ratio_net = 2)
{
    std::vector<cv::RotatedRect> adjusted_polys;

    for (const auto &poly : polys)
    {
        cv::Point2f corners[4];
        poly.points(corners);

        for (int i = 0; i < 4; ++i)
        {
            corners[i].x *= (ratio_w * ratio_net);
            corners[i].y *= (ratio_h * ratio_net);
        }

        // cv::RotatedRect adjusted_rect(cv::Point2f(0, 0), cv::Size2f(0, 0), 0);
        cv::RotatedRect adjusted_rect = cv::minAreaRect(std::vector<cv::Point2f>(corners, corners + 4));
        // adjusted_rect.points(corners);
        adjusted_polys.push_back(adjusted_rect);
    }

    return adjusted_polys;
}

int main(int argc, const char *argv[])
{
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
    cv::Mat image_original = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error reading image from file\n";
        return -1;
    }

    // torch.Size([1, 3, 672, 1248])
    // cv::resize(image, image, cv::Size(312 * 2, 168 * 2));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // cv::Mat img_cv_grey;
    // cv::cvtColor(image, img_cv_grey, cv::COLOR_BGR2GRAY);

    float canvas_size = 2560;
    float mag_ratio = 1.0;

    cv::Mat image_resized;
    float target_ratio;
    cv::Size size_heatmap; // TODO - can probably get rid of this size_heatmap var
    std::tie(image_resized, target_ratio, size_heatmap) = resize_aspect_ratio(
        image, canvas_size, cv::INTER_LINEAR, mag_ratio);

    float ratio_h = 1 / target_ratio;
    float ratio_w = 1 / target_ratio;

    torch::Tensor image_tensor = torch::from_blob(
        image_resized.data, {1, image_resized.rows, image_resized.cols, 3}, torch::kByte);

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

            // display_2d_tensor_heatmap("main score_text", score_text);
            // display_2d_tensor_heatmap("main score_link", score_link);

            // Set parameters for the function
            float text_threshold = 0.7;
            float link_threshold = 0.4;
            float low_text = 0.4;
            bool estimate_num_chars = false;
            auto result = get_detected_boxes(score_text, score_link, text_threshold, link_threshold, low_text, estimate_num_chars);

            // Extract results
            auto det = result.first;
            auto labels = result.second;

            // std::cout << "Box before adjustment: " << det.size() << std::endl;
            // for (const auto &box : det)
            // {
            //     cv::Point2f corners[4];
            //     box.points(corners);
            //     std::cout << "Box BA: ";
            //     for (int i = 0; i < 4; ++i)
            //     {
            //         std::cout << "(" << corners[i].x << ", " << corners[i].y << ") ";
            //     }
            //     std::cout << std::endl;
            // }

            // Scale bounding boxes to the input image * ratio
            auto boxes = adjust_result_coordinates(det, ratio_w, ratio_h);

            // Draw the detected boxes on the image
            // for (const cv::RotatedRect &box : boxes)
            // {
            //     cv::Point2f corners[4];
            //     box.points(corners);
            //     std::vector<cv::Point> corners_vec(corners, corners + 4);
            //     cv::polylines(image, corners_vec, true, cv::Scalar(0, 255, 0), 2);
            // }

            // // Display the image with the drawn boxes
            // cv::namedWindow("Detected Boxes", cv::WINDOW_NORMAL);
            // cv::imshow("Detected Boxes", image);
            // cv::waitKey(0);

            // Create a vector to store the cropped images
            cv::Mat all_cropped_images = cv::Mat::zeros(image.size(), image.type());
            std::vector<std::pair<cv::RotatedRect, cv::Mat>> text_regions;
            for (const cv::RotatedRect &box : boxes)
            {
                // Crop the rotated image using the bounding rectangle
                // TODO - probably want to group the boxes horizontal if they are within some distance of each other
                // this will probably give better results for the transformer model reading the text as there will be
                // more context to infer the letters from. It will also mean less forward passes through the model.
                cv::Mat cropped_image = image(box.boundingRect());
                cropped_image.copyTo(all_cropped_images(box.boundingRect()));
                text_regions.push_back(std::make_pair(box, cropped_image));
            }

            // Save all_cropped_images
            cv::imwrite("/Users/jackvial/Code/CPlusPlus/torchscript_example/outputs/all_cropped_images.jpg", all_cropped_images);

            // ==== Recognition Stage ====
            std::string parseq_model_path = "/Users/jackvial/Code/CPlusPlus/torchscript_example/parseq-tiny/torchscript_model.bin";

            // Deserialize the TorchScript module from a file
            torch::jit::script::Module parseq_model;
            try
            {
                parseq_model = torch::jit::load(parseq_model_path);
            }
            catch (const c10::Error &e)
            {
                std::cerr << "error loading the parseq model\n";
                return -1;
            }

            std::cout << "parseq model loaded\n";

            for (auto &text_region : text_regions)
            {
                cv::Mat parseq_image_input;
                cv::resize(text_region.second, parseq_image_input, cv::Size(128, 32));
                cv::cvtColor(parseq_image_input, parseq_image_input, cv::COLOR_BGR2RGB);

                torch::Tensor parseq_image_tensor = torch::from_blob(
                    parseq_image_input.data, {1, parseq_image_input.rows, parseq_image_input.cols, 3}, torch::kByte);

                parseq_image_tensor = parseq_image_tensor.permute({0, 3, 1, 2}); // Rearrange dimensions to {1, 3, 32, 128}
                parseq_image_tensor = parseq_image_tensor.to(torch::kFloat);
                parseq_image_tensor = parseq_image_tensor.div(255.0); // Normalize pixel values (0-255 -> 0-1)

                // Create a vector of inputs
                std::vector<torch::jit::IValue> parseq_inputs;
                // parseq_inputs.push_back(torch::ones({1, 3, 32, 128}));
                parseq_inputs.push_back(parseq_image_tensor);

                // Execute the model and turn its output into a tensor
                at::Tensor parseq_output = parseq_model.forward(parseq_inputs).toTensor();
                print_tensor_dims(" parseq_output ", parseq_output);
            }
        }
    }
    else
    {
        std::cerr << "Model output is not a tuple\n";
        return -1;
    }

    return 0;
}
