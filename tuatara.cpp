#include "tuatara.h"

#include <torch/script.h>
#include <torch/version.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "utils.h"

using namespace torch::indexing;

class Tokenizer {
 public:
  const char BOS = '[';
  const char EOS = ']';
  const char PAD = 'P';

  Tokenizer() {
    const std::string charset =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&"
        "\\'()*+,-./:;<=>?@[\\]^_`{|}~";

    itos = charset;
    itos.insert(itos.begin(), EOS);
    itos.push_back(BOS);
    itos.push_back(PAD);

    for (size_t i = 0; i < itos.size(); ++i) {
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
   *   token_dists: softmax probabilities over the token distribution. Shape: N,
   * L, C raw: return unprocessed labels (will return list of list of strings)
   *
   * Returns:
   *   A list of string labels (arbitrary length) and
   *   their corresponding sequence probabilities as a list of vectors.
   */
  std::vector<std::string> decode(const torch::Tensor &token_dists, bool raw = false) {
    std::vector<std::string> batch_tokens;
    for (int64_t i = 0; i < token_dists.size(0); ++i) {
      auto dist = token_dists[i];
      auto max_dist_result = max_dist(dist);
      torch::Tensor probs = std::get<0>(max_dist_result);
      torch::Tensor ids = std::get<1>(max_dist_result);

      if (!raw) {
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

  std::vector<size_t> tok2ids(const std::string &tokens) {
    std::vector<size_t> ids;
    for (char s : tokens) {
      ids.push_back(stoi[s]);
    }
    return ids;
  }

  std::string ids2tok(const std::vector<size_t> &token_ids, bool join = true) {
    std::string tokens;
    for (size_t id : token_ids) {
      tokens.push_back(itos[id]);
    }
    return tokens;
  }

  std::pair<torch::Tensor, torch::Tensor> max_dist(const torch::Tensor &dist) {
    torch::Tensor probs, ids;
    std::tie(probs, ids) = dist.max(-1);

    return std::make_pair(probs, ids);
  }

  std::tuple<torch::Tensor, torch::Tensor> filter(const torch::Tensor &probs, const torch::Tensor &ids) {
    torch::Tensor filtered_probs, filtered_ids;
    auto eos_mask = (ids != c10::Scalar(static_cast<int64_t>(eos_id)));

    filtered_probs = probs.masked_select(eos_mask);
    filtered_ids = ids.masked_select(eos_mask);

    return std::make_tuple(filtered_probs, filtered_ids);
  }
};

std::pair<std::vector<cv::RotatedRect>, cv::Mat> get_detected_boxes(torch::Tensor textmap, torch::Tensor linkmap, float text_threshold, float link_threshold, float low_text, bool estimate_num_chars = false) {
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
  for (int k = 1; k < nLabels; ++k) {
    int size = stats.at<int>(k, cv::CC_STAT_AREA);
    if (size < 10) continue;

    cv::Mat mask = (labels == k);
    double minVal, maxVal;
    cv::minMaxLoc(textmap_cv, &minVal, &maxVal, nullptr, nullptr, mask);

    if (maxVal < text_threshold) continue;

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

    if (std::abs(1 - box_ratio) <= 0.1) {
      int l = std::min_element(np_contours.begin(), np_contours.end(), [](const cv::Point &a, const cv::Point &b) { return a.x < b.x; })->x;
      int r = std::max_element(np_contours.begin(), np_contours.end(), [](const cv::Point &a, const cv::Point &b) { return a.x < b.x; })->x;
      int t = std::min_element(np_contours.begin(), np_contours.end(), [](const cv::Point &a, const cv::Point &b) { return a.y < b.y; })->y;
      int b = std::max_element(np_contours.begin(), np_contours.end(), [](const cv::Point &a, const cv::Point &b) { return a.y < b.y; })->y;

      cv::Point2f new_box[4] = {cv::Point2f(l, t), cv::Point2f(r, t), cv::Point2f(r, b), cv::Point2f(l, b)};
      std::copy(new_box, new_box + 4, box);
    }

    int startidx = std::distance(box, std::min_element(box, box + 4, [](const cv::Point2f &a, const cv::Point2f &b) { return a.x + a.y < b.x + b.y; }));
    std::rotate(box, box + startidx, box + 4);

    det.emplace_back(rectangle);
  }

  return std::make_pair(det, labels);
}

std::tuple<cv::Mat, float, cv::Size> resize_aspect_ratio(const cv::Mat &img, int square_size, int interpolation, float mag_ratio = 1) {
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  float target_size = mag_ratio * std::max(height, width);

  if (target_size > square_size) {
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

std::vector<cv::RotatedRect> adjust_result_coordinates(const std::vector<cv::RotatedRect> &polys, float ratio_w, float ratio_h, float ratio_net = 2) {
  std::vector<cv::RotatedRect> adjusted_polys;

  for (const auto &poly : polys) {
    cv::Point2f corners[4];
    poly.points(corners);

    for (int i = 0; i < 4; ++i) {
      corners[i].x *= (ratio_w * ratio_net);
      corners[i].y *= (ratio_h * ratio_net);
    }

    cv::RotatedRect adjusted_rect = cv::minAreaRect(std::vector<cv::Point2f>(corners, corners + 4));
    adjusted_polys.push_back(adjusted_rect);
  }

  return adjusted_polys;
}

// @TODO - change bbox output to ints
std::vector<float> rotated_rect_to_tesseract_format(const cv::RotatedRect &rect) {
  cv::Point2f vertices[4];
  rect.points(vertices);

  // Finding the top-left, bottom-right points
  float min_x = std::min(std::min(vertices[0].x, vertices[1].x), std::min(vertices[2].x, vertices[3].x));
  float min_y = std::min(std::min(vertices[0].y, vertices[1].y), std::min(vertices[2].y, vertices[3].y));
  float max_x = std::max(std::max(vertices[0].x, vertices[1].x), std::max(vertices[2].x, vertices[3].x));
  float max_y = std::max(std::max(vertices[0].y, vertices[1].y), std::max(vertices[2].y, vertices[3].y));
  
  
  min_x = std::round(min_x);
  min_y = std::round(min_y);
  max_x = std::round(max_x);
  max_y = std::round(max_y);

  std::vector<float> result = {min_x, min_y, max_x, max_y};
  return result;
}

std::vector<OutputItem> format_output(const std::vector<std::pair<std::string, cv::RotatedRect>> &predicted_text_bbox_pairs) {
  std::vector<OutputItem> formatted_output;

  for (const auto &pair : predicted_text_bbox_pairs) {
    OutputItem output_item;
    output_item.text = pair.first;
    output_item.bbox = rotated_rect_to_tesseract_format(pair.second);
    formatted_output.push_back(output_item);
  }

  return formatted_output;
}

void infer(torch::jit::script::Module &model, std::queue<std::pair<int, torch::Tensor>> &input_queue, std::vector<std::pair<int, torch::Tensor>> &outputs, std::mutex &input_mutex, std::mutex &output_mutex) {
  torch::NoGradGuard no_grad;

  while (true) {
    input_mutex.lock();
    if (input_queue.empty()) {
      input_mutex.unlock();
      break;
    }

    int input_idx = input_queue.front().first;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_queue.front().second);

    input_queue.pop();
    input_mutex.unlock();

    torch::Tensor output = model.forward(inputs).toTensor();

    std::unique_lock<std::mutex> lock(output_mutex);
    outputs.push_back(std::make_pair(input_idx, output));
  }
}

std::vector<OutputItem> image_to_data(cv::Mat image, std::string weights_dir, std::string outputs_dir) {
  if (weights_dir.empty()) {
    std::cerr << "Please provide a value for weights_dir" << std::endl;
    return {};
  }

  if (outputs_dir.empty()) {
    std::cerr << "Please provide a value for outputs_dir" << std::endl;
    return {};
  }

  // Disable gradient calculation
  // This reduces memory usage by ~10x
  torch::NoGradGuard no_grad;
  std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  std::string model_path = weights_dir + "/craft_traced_torchscript_model.pt";
  torch::jit::script::Module detector_model;
  try {
    detector_model = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading craft model";
    return {};
  }

  std::cout << "craft model loaded" << std::endl;

  if (image.empty()) {
    std::cerr << "Error reading image from file";
    return {};
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  // This has a big effect on memory and ability to detect boxes
  float canvas_size = 1024;
  float mag_ratio = 1.0;

  cv::Mat image_resized;
  float target_ratio;
  cv::Size size_heatmap;  // TODO - can probably get rid of this size_heatmap var
  std::tie(image_resized, target_ratio, size_heatmap) = resize_aspect_ratio(image, canvas_size, cv::INTER_LINEAR, mag_ratio);

  float ratio_h = 1 / target_ratio;
  float ratio_w = 1 / target_ratio;

  torch::Tensor image_tensor = torch::from_blob(image_resized.data, {1, image_resized.rows, image_resized.cols, 3}, torch::kByte);

  // Rearrange dimensions to {1, 3, 32, 128}
  image_tensor = image_tensor.permute({0, 3, 1, 2});
  image_tensor = image_tensor.to(torch::kFloat);

  // Normalize pixel values (0-255 -> 0-1)
  image_tensor = image_tensor.div(255.0);

  std::vector<torch::jit::IValue> detector_inputs;
  detector_inputs.push_back(image_tensor);

  // Execute the model and turn its output into a tensor
  torch::jit::IValue detector_output = detector_model.forward(detector_inputs);
  if (!detector_output.isTuple()) {
    std::cerr << "Model output is not a tuple\n";
    return {};
  }

  auto detector_output_tuple = detector_output.toTuple();

  torch::Tensor detector_output_1 = detector_output_tuple->elements()[0].toTensor();

  std::cout << "post processing craft predictions..." << std::endl;
  int64_t batch_size = detector_output_1.size(0);

  // Assume only one batch is ever passed to the detector model
  torch::Tensor current_batch = detector_output_1[0];

  // Separate the tensor into two tensors of dimensions 624 x 336
  torch::Tensor score_text = current_batch.slice(2, 0, 1).squeeze(2);
  torch::Tensor score_link = current_batch.slice(2, 1, 2).squeeze(2);

  // TODO - move these to a config struct
  float text_threshold = 0.7;
  float link_threshold = 0.4;
  float low_text = 0.4;
  auto result = get_detected_boxes(score_text, score_link, text_threshold, link_threshold, low_text);

  auto det = result.first;
  auto labels = result.second;

  // Scale bounding boxes to the input image * ratio
  auto boxes = adjust_result_coordinates(det, ratio_w, ratio_h);

  std::vector<std::pair<cv::RotatedRect, cv::Mat>> text_regions;
  for (const cv::RotatedRect &box : boxes) {
    // Crop the rotated image using the bounding rectangle
    // TODO - probably want to group the boxes horizontal if they are within
    // some distance of each other this will probably give better results for
    // the transformer model reading the text as there will be more context to
    // infer the letters from. It will also mean less forward passes through
    // the model.
    cv::Mat cropped_image = image(box.boundingRect());
    text_regions.push_back(std::make_pair(box, cropped_image));
  }

  // ==== Recognition Stage ====
  std::cout << "loading parseq model..." << std::endl;

  std::string parseq_model_path = weights_dir + "/parseq_torchscript.bin";

  // Deserialize the TorchScript module from a file
  torch::jit::script::Module parseq_model;
  try {
    parseq_model = torch::jit::load(parseq_model_path);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the parseq model\n";
    return {};
  }

  std::cout << "parseq model loaded\n";

  // Transformer text regions into tensors for parseq model
  std::vector<torch::Tensor> parseq_tensors;
  for (auto &text_region : text_regions) {
    cv::Mat parseq_image_input;
    cv::resize(text_region.second, parseq_image_input, cv::Size(128, 32));
    cv::cvtColor(parseq_image_input, parseq_image_input, cv::COLOR_BGR2RGB);

    torch::Tensor parseq_tensor = torch::from_blob(parseq_image_input.data, {1, parseq_image_input.rows, parseq_image_input.cols, 3}, torch::kByte);
    parseq_tensor = parseq_tensor.permute({0, 3, 1, 2});
    parseq_tensor = parseq_tensor.to(torch::kFloat);
    parseq_tensor = parseq_tensor.div(255.0);
    parseq_tensors.push_back(parseq_tensor);
  }

  std::queue<std::pair<int, torch::Tensor>> input_queue;

  int chunk_size = 4;
  for (size_t i = 0; i < parseq_tensors.size(); i += chunk_size) {
    // Create a new chunk using elements from the current index to the next n elements
    std::vector<torch::Tensor> chunk(parseq_tensors.begin() + i, parseq_tensors.begin() + std::min(i + chunk_size, parseq_tensors.size()));

    // Add the chunk to the result vector
    input_queue.push(std::make_pair(i, torch::cat(chunk, 0)));
  }

  const int num_threads = 6;

  std::vector<std::thread> threads;
  std::vector<std::pair<int, torch::Tensor>> parseq_outputs;
  std::mutex input_mutex, output_mutex;

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(infer, std::ref(parseq_model), std::ref(input_queue), std::ref(parseq_outputs), std::ref(input_mutex), std::ref(output_mutex));
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Sort the outputs based on their input indices
  std::sort(parseq_outputs.begin(), parseq_outputs.end(), [](const std::pair<int, torch::Tensor> &a, const std::pair<int, torch::Tensor> &b) { return a.first < b.first; });

  std::vector<torch::Tensor> sorted_outputs;
  for (const auto &output : parseq_outputs) {
    sorted_outputs.push_back(output.second);
  }

  torch::Tensor parseq_output_tensor = torch::cat(sorted_outputs, 0);
  auto parseq_pred = torch::softmax(parseq_output_tensor, -1);

  std::cout << "Running tokenizer..." << std::endl;

  std::vector<std::pair<std::string, cv::RotatedRect>> predicted_text_bbox_pairs;

  Tokenizer tokenizer;
  std::vector<std::string> tokens = tokenizer.decode(parseq_pred, false);
  std::size_t tokens_size = tokens.size();
  for (int64_t token_item_index = 0; token_item_index < tokens_size; ++token_item_index) {
    std::string predicted_text;
    for (const auto &token_char : tokens[token_item_index]) {
      if (token_char == tokenizer.EOS) {
        break;
      }
      predicted_text.push_back(token_char);
    }

    predicted_text_bbox_pairs.push_back(std::make_pair(predicted_text, text_regions[token_item_index].first));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  std::cout << "Elapsed time: " << (elapsed_time * 0.001) << " seconds " << std::endl;

  return format_output(predicted_text_bbox_pairs);
}
