#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <vector>

void draw_bounding_boxes_on_background(
    const std::vector<cv::RotatedRect> &boxes);
void print_tensor_dims(std::string label, torch::Tensor t);
void display_2d_tensor_heatmap(std::string label, torch::Tensor t);

#endif // UTILS_H
