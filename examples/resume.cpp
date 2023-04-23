#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tuatara.h"

// Main file
int main(int argc, const char** argv) {
  std::string image_path = argv[1];
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

  image_to_data(image, argv[2], argv[3]);
  return 0;
}