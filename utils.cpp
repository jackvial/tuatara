#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

void draw_bounding_boxes_on_background(
    const std::vector<cv::RotatedRect> &boxes) {
  // Find the enclosing background size
  float min_x = FLT_MAX, min_y = FLT_MAX, max_x = FLT_MIN, max_y = FLT_MIN;

  for (const auto &box : boxes) {
    cv::Point2f corners[4];
    box.points(corners);

    for (int i = 0; i < 4; ++i) {
      min_x = std::min(min_x, corners[i].x);
      min_y = std::min(min_y, corners[i].y);
      max_x = std::max(max_x, corners[i].x);
      max_y = std::max(max_y, corners[i].y);
    }
  }

  int background_width = static_cast<int>(max_x - min_x) + 1;
  int background_height = static_cast<int>(max_y - min_y) + 1;

  // Create a new image (background) with the enclosing size
  cv::Mat background =
      cv::Mat::zeros(cv::Size(background_width, background_height), CV_8UC3);

  // Draw the bounding boxes on the background
  for (const auto &box : boxes) {
    cv::Point2f corners[4];
    box.points(corners);

    // Shift corner points to the new coordinate system of the background image
    for (int i = 0; i < 4; ++i) {
      corners[i].x -= min_x;
      corners[i].y -= min_y;
    }

    std::vector<cv::Point> corners_vec(corners, corners + 4);
    cv::polylines(background, corners_vec, true, cv::Scalar(0, 255, 0), 2);
  }

  // Display the background image with bounding boxes
  cv::imshow("Bounding Boxes", background);
  cv::waitKey(0);
}

void print_tensor_dims(std::string label, torch::Tensor t) {
  int64_t num_dims = t.dim();
  // Print the dimensions of the tensor
  std::cout << label << " (";
  for (int64_t i = 0; i < num_dims; ++i) {
    std::cout << t.size(i);
    if (i < num_dims - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ")" << std::endl;
}

void display_2d_tensor_heatmap(std::string label, torch::Tensor t) {
  // Normalize the tensor to the range [0, 1]
  torch::Tensor tensor_normalized = (t - t.min()) / (t.max() - t.min());

  // Convert the normalized tensor to an OpenCV Mat
  cv::Mat mat(tensor_normalized.size(0), tensor_normalized.size(1), CV_32F,
              tensor_normalized.data_ptr<float>());

  // Convert the normalized Mat to a heatmap
  cv::Mat heatmap;
  mat.convertTo(heatmap, CV_8UC1, 255);
  cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);

  // Display the heatmap
  cv::imshow(label, heatmap);
  cv::waitKey(0);
}