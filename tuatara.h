#ifndef TUATARA_H
#define TUATARA_H
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

int image_to_data(cv::Mat image, std::string weights_dir, std::string outputs_dir, std::string debug_mode);

#endif  // TUATARA_H
