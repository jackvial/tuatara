#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tuatara.h"

namespace py = pybind11;

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
  py::buffer_info buf1 = input1.request(), buf2 = input2.request();

  if (buf1.ndim != 1 || buf2.ndim != 1) throw std::runtime_error("Number of dimensions must be one");

  if (buf1.size != buf2.size) throw std::runtime_error("Input shapes must match");

  /* No pointer is passed, so NumPy will allocate the buffer */
  auto result = py::array_t<double>(buf1.size);

  py::buffer_info buf3 = result.request();

  double *ptr1 = static_cast<double *>(buf1.ptr);
  double *ptr2 = static_cast<double *>(buf2.ptr);
  double *ptr3 = static_cast<double *>(buf3.ptr);

  for (size_t idx = 0; idx < buf1.shape[0]; idx++) ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}

cv::Mat apply_sobel(cv::Mat img, int ddepth, int dx, int dy, int ksize) {
  cv::Mat gray_img, result;

  // Convert the input image to grayscale
  cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

  // Apply the Sobel filter
  cv::Sobel(gray_img, result, ddepth, dx, dy, ksize);

  double min_val, max_val;
  cv::minMaxLoc(result, &min_val, &max_val);
  result.convertTo(result, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

  return result;
}

cv::Mat buffer_to_mat(py::array_t<unsigned char> input) {
  py::buffer_info buf = input.request();

  if (buf.ndim != 3) {
    throw std::runtime_error("Input array should have 3 dimensions");
  }

  int rows = buf.shape[0];
  int cols = buf.shape[1];
  int channels = buf.shape[2];

  cv::Mat mat(rows, cols, CV_8UC3);

  std::memcpy(mat.data, buf.ptr, mat.total() * mat.elemSize());

  return mat;
}

py::array_t<unsigned char> mat_to_buffer(cv::Mat mat) {
  py::array_t<unsigned char> result(mat.rows * mat.cols * mat.channels(), mat.data);
  result.resize({mat.rows, mat.cols, mat.channels()});
  return result;
}

py::array_t<unsigned char> apply_sobel_filter(py::array_t<unsigned char> input, int ddepth, int dx, int dy, int ksize) {
  cv::Mat img = buffer_to_mat(input);
  cv::Mat sobel_img = apply_sobel(img, ddepth, dx, dy, ksize);
  return mat_to_buffer(sobel_img);
}

py::array_t<unsigned char> image_to_data_bindings(
    py::array_t<unsigned char> image_data,
    std::string weights_path,
    std::string output_path,
    std::string debug
  ) {
  cv::Mat img = buffer_to_mat(image_data);

  // cv::Mat result;
  // double min_val, max_val;
  // cv::minMaxLoc(img, &min_val, &max_val);
  // result.convertTo(result, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
  image_to_data(img, weights_path, output_path, debug);

  return mat_to_buffer(img);
}

PYBIND11_MODULE(pytuatara, m) {
  m.doc() = "Tuatara ocr";

  m.def("image_to_data", &image_to_data_bindings, "Runs OCR on given image path and saves to given output path");

//   m.def("add_arrays", &add_arrays, "Add two NumPy arrays");

//   m.def("test_input_output", &image_to_data_bindings, "Test input and output");
}
