#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <cmath>
#include <utility>
#include <torch/script.h>
#include <torch/version.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"

using namespace torch::indexing;

class Tokenizer
{
public:
    const char BOS = '[';
    const char EOS = ']';
    const char PAD = 'P';

    Tokenizer()
    {
        const std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\\'()*+,-./:;<=>?@[\\]^_`{|}~";

        itos = charset;
        itos.insert(itos.begin(), EOS);
        itos.push_back(BOS);
        itos.push_back(PAD);

        for (size_t i = 0; i < itos.size(); ++i)
        {
            stoi[itos[i]] = i;
        }

        eos_id = stoi[EOS];
        bos_id = stoi[BOS];
        pad_id = stoi[PAD];
    }

    /**
     * Decode a batch of token distributions.
     *
     * Args:
     *   token_dists: softmax probabilities over the token distribution. Shape: N, L, C
     *   raw: return unprocessed labels (will return list of list of strings)
     *
     * Returns:
     *   A list of string labels (arbitrary length) and
     *   their corresponding sequence probabilities as a list of vectors.
     */
    std::vector<std::string> decode(const torch::Tensor &token_dists, bool raw = false)
    {
        std::vector<std::string> batch_tokens;
        for (int64_t i = 0; i < token_dists.size(0); ++i)
        {
            auto dist = token_dists[i];
            auto max_dist_result = max_dist(dist);
            torch::Tensor probs = std::get<0>(max_dist_result);
            torch::Tensor ids = std::get<1>(max_dist_result);

            if (!raw)
            {
                std::tie(probs, ids) = filter(probs, ids);
            }

            std::vector<size_t> id_vector(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
            std::string tokens = ids2tok(id_vector, !raw);
            batch_tokens.push_back(tokens);
        }
        return batch_tokens;
    }

private:
    std::string itos;
    std::map<char, size_t> stoi;
    size_t eos_id, bos_id, pad_id;

    std::vector<size_t> tok2ids(const std::string &tokens)
    {
        std::vector<size_t> ids;
        for (char s : tokens)
        {
            ids.push_back(stoi[s]);
        }
        return ids;
    }

    std::string ids2tok(const std::vector<size_t> &token_ids, bool join = true)
    {
        std::string tokens;
        for (size_t id : token_ids)
        {
            tokens.push_back(itos[id]);
        }
        return tokens;
    }

    std::pair<torch::Tensor, torch::Tensor> max_dist(const torch::Tensor &dist)
    {
        torch::Tensor probs, ids;
        std::tie(probs, ids) = dist.max(-1);

        return std::make_pair(probs, ids);
    }

    std::tuple<torch::Tensor, torch::Tensor> filter(const torch::Tensor &probs, const torch::Tensor &ids)
    {
        torch::Tensor filtered_probs, filtered_ids;
        auto eos_mask = (ids != c10::Scalar(static_cast<int64_t>(eos_id)));

        filtered_probs = probs.masked_select(eos_mask);
        filtered_ids = ids.masked_select(eos_mask);

        return std::make_tuple(filtered_probs, filtered_ids);
    }
};

std::pair<std::vector<cv::RotatedRect>, cv::Mat> get_detected_boxes(
    torch::Tensor textmap, torch::Tensor linkmap, float text_threshold,
    float link_threshold, float low_text, bool estimate_num_chars = false)
{
    torch::Tensor textmap_normalized = (textmap - textmap.min()) / (textmap.max() - textmap.min());
    torch::Tensor linkmap_normalized = (linkmap - linkmap.min()) / (linkmap.max() - linkmap.min());

    cv::Mat textmap_cv(textmap_normalized.size(0), textmap_normalized.size(1), CV_32F, textmap_normalized.data_ptr<float>());
    cv::Mat linkmap_cv(linkmap_normalized.size(0), linkmap_normalized.size(1), CV_32F, linkmap_normalized.data_ptr<float>());

    int img_h = textmap_cv.rows;
    int img_w = textmap_cv.cols;

    // Threshold to binary image for connect component labeling
    cv::Mat text_score, link_score;
    cv::threshold(textmap_cv, text_score, low_text, 1, 0);
    cv::threshold(linkmap_cv, link_score, link_threshold, 1, 0);

    // Combine both threshold images. Worth experimenting to see
    // if using either text_score or link_score provides a better score
    cv::Mat text_score_comb = cv::min(cv::max(text_score + link_score, 0.0), 1.0);
    text_score_comb.convertTo(text_score_comb, CV_8U);

    // Connect component labeling
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
        mapper.push_back(k);

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

        // Find the min bounding rect of the contour
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

        cv::RotatedRect adjusted_rect = cv::minAreaRect(std::vector<cv::Point2f>(corners, corners + 4));
        adjusted_polys.push_back(adjusted_rect);
    }

    return adjusted_polys;
}

std::vector<std::vector<std::pair<cv::RotatedRect, cv::Mat>>> make_recognizer_model_batches(const std::vector<std::pair<cv::RotatedRect, cv::Mat>> &input, int n)
{
    std::vector<std::vector<std::pair<cv::RotatedRect, cv::Mat>>> chunks;

    int chunk_size = input.size() / n;
    int remainder = input.size() % n;

    int start = 0;
    for (int i = 0; i < n; ++i)
    {
        int end = start + chunk_size + (i < remainder ? 1 : 0);
        std::vector<std::pair<cv::RotatedRect, cv::Mat>> chunk(input.begin() + start, input.begin() + end);
        chunks.push_back(chunk);
        start = end;
    }

    return chunks;
}

std::string escape_json_string(const std::string &s)
{
    std::ostringstream o;
    for (auto c : s)
    {
        if (c == '"' || c == '\\')
        {
            o << '\\';
        }
        o << c;
    }
    return o.str();
}

std::string rotated_rect_to_tesseract_format(const cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];
    rect.points(vertices);

    // Finding the top-left, bottom-right points
    float min_x = std::min(std::min(vertices[0].x, vertices[1].x), std::min(vertices[2].x, vertices[3].x));
    float min_y = std::min(std::min(vertices[0].y, vertices[1].y), std::min(vertices[2].y, vertices[3].y));
    float max_x = std::max(std::max(vertices[0].x, vertices[1].x), std::max(vertices[2].x, vertices[3].x));
    float max_y = std::max(std::max(vertices[0].y, vertices[1].y), std::max(vertices[2].y, vertices[3].y));

    std::stringstream ss;
    ss << "[" << min_x << "," << min_y << "," << max_x << "," << max_y << "]";
    return ss.str();
}

std::string to_json(const std::vector<std::pair<std::string, cv::RotatedRect>> &predicted_text_bbox_pairs)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < predicted_text_bbox_pairs.size(); ++i)
    {
        const auto &pair = predicted_text_bbox_pairs[i];
        ss << "{\"text\": \"" << escape_json_string(pair.first) << "\", "
           << "\"bbox\": " << rotated_rect_to_tesseract_format(pair.second) << "}";
        if (i < predicted_text_bbox_pairs.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

void save_to_file(const std::string &filename, const std::string &content)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << content;
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

std::string get_file_name_from_path(const std::string &image_path)
{
    std::size_t start = image_path.find_last_of('/') + 1;
    std::size_t end = image_path.find_last_of('.');
    return image_path.substr(start, end - start);
}

int main(int argc, const char *argv[])
{
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    std::string model_path = "../weights/craft_traced_torchscript_model.pt";

    // Deserialize the TorchScript module from a file
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading craft model\n";
        return -1;
    }

    std::cout << "craft model loaded\n";

    std::string image_path = "../images/resume_example.png";
    std::string image_file_name = get_file_name_from_path(image_path);
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat image_original = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error reading image from file\n";
        return -1;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    float canvas_size = 1024; // This has a big effect on memory and ability to detect boxes
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
    if (!output_ivalue.isTuple())
    {
        std::cerr << "Model output is not a tuple\n";
        return -1;
    }

    auto output_tuple = output_ivalue.toTuple();

    // Access elements in the tuple using std::get
    torch::Tensor output_tensor_1 = output_tuple->elements()[0].toTensor();
    torch::Tensor output_tensor_2 = output_tuple->elements()[1].toTensor();

    // Get the size of the batch dimension
    int64_t batch_size = output_tensor_1.size(0);

    // Iterate through the batch dimension
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        // Access the tensor corresponding to the current batch
        torch::Tensor current_batch = output_tensor_1[batch_index];

        // Separate the tensor into two tensors of dimensions 624 x 336
        torch::Tensor score_text = current_batch.slice(2, 0, 1).squeeze(2);
        torch::Tensor score_link = current_batch.slice(2, 1, 2).squeeze(2);

        // Set parameters for the function
        float text_threshold = 0.7;
        float link_threshold = 0.4;
        float low_text = 0.4;
        auto result = get_detected_boxes(score_text, score_link, text_threshold, link_threshold, low_text);

        auto det = result.first;
        auto labels = result.second;

        // Scale bounding boxes to the input image * ratio
        auto boxes = adjust_result_coordinates(det, ratio_w, ratio_h);

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
        cv::imwrite("../outputs/" + image_file_name + "_detector_crops.jpg", all_cropped_images);

        // ==== Recognition Stage ====
        std::string parseq_model_path = "../weights/parseq_torchscript.bin";
        // std::string parseq_model_path = "../weights/parseq_int8_torchscript.pt";

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

        std::vector<std::pair<std::string, cv::RotatedRect>> predicted_text_bbox_pairs;
        std::vector<std::vector<std::pair<cv::RotatedRect, cv::Mat>>> batches = make_recognizer_model_batches(text_regions, 4);
        std::size_t n_parseq_batches = batches.size();
        std::size_t parseq_batch_offset = 0;
        for (int64_t parseq_batch_index = 0; parseq_batch_index < n_parseq_batches; ++parseq_batch_index)
        {
            std::vector<torch::Tensor> parseq_batch;
            for (auto &text_region : batches[parseq_batch_index])
            {
                cv::Mat parseq_image_input;
                cv::resize(text_region.second, parseq_image_input, cv::Size(128, 32));
                cv::cvtColor(parseq_image_input, parseq_image_input, cv::COLOR_BGR2RGB);

                torch::Tensor parseq_image_tensor = torch::from_blob(
                    parseq_image_input.data, {1, parseq_image_input.rows, parseq_image_input.cols, 3}, torch::kByte);
                parseq_image_tensor = parseq_image_tensor.permute({0, 3, 1, 2}); // Rearrange dimensions to {1, 3, 32, 128}
                parseq_image_tensor = parseq_image_tensor.to(torch::kFloat);
                parseq_image_tensor = parseq_image_tensor.div(255.0); // Normalize pixel values (0-255 -> 0-1)
                parseq_batch.push_back(parseq_image_tensor);
                predicted_text_bbox_pairs.push_back(std::make_pair("", text_region.first)); // Set placeholder text
            }

            // This is the list of arguments to the model forward pass e.g. model.forward(x, y)
            // in most cases this will only be a list of length 1 e.g. model.forward(x)
            // Don't confuse it with the batches.
            std::vector<torch::jit::IValue> parseq_inputs;
            torch::Tensor concat_batch = torch::cat(parseq_batch, 0);

            parseq_inputs.push_back(concat_batch);
            at::Tensor parseq_output = parseq_model.forward(parseq_inputs).toTensor();

            auto parseq_pred = torch::softmax(parseq_output, -1);

            Tokenizer tokenizer;
            std::vector<std::string> token_batch = tokenizer.decode(parseq_pred, false);
            std::size_t token_batch_size = token_batch.size();
            for (int64_t token_item_index = 0; token_item_index < token_batch_size; ++token_item_index)
            {
                std::string predicted_text;
                for (const auto &token_char : token_batch[token_item_index])
                {
                    if (token_char == tokenizer.EOS)
                    {
                        break;
                    }
                    predicted_text.push_back(token_char);
                }

                predicted_text_bbox_pairs[parseq_batch_offset + token_item_index].first = predicted_text;
            }
            parseq_batch_offset += token_batch_size;
        }

        std::size_t n_predicted_pairs = predicted_text_bbox_pairs.size();
        for (int64_t pred_pair_index = 0; pred_pair_index < n_predicted_pairs; ++pred_pair_index)
        {
            // Draw the detected boxes on the image
            cv::Point2f corners[4];
            predicted_text_bbox_pairs[pred_pair_index].second.points(corners);
            std::vector<cv::Point> corners_vec(corners, corners + 4);
            cv::polylines(image, corners_vec, true, cv::Scalar(0, 255, 0), 2);

            // Draw the text inside the bounding box
            const std::string &predicted_text = predicted_text_bbox_pairs[pred_pair_index].first;
            cv::Point text_origin(corners[0].x, corners[0].y); // You can adjust the position as needed
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int font_thickness = 2;
            cv::Scalar text_color(0, 0, 255); // BGR color: red
            cv::putText(image, predicted_text, text_origin, font_face, font_scale, text_color, font_thickness);
        }

        // Display the image with the drawn boxes
        cv::namedWindow("Results", cv::WINDOW_NORMAL);
        cv::imshow("Results", image);
        cv::waitKey(0);

        std::string json_str = to_json(predicted_text_bbox_pairs);
        save_to_file("../outputs/" + image_file_name + "_results.json", json_str);
    }

    return 0;
}
