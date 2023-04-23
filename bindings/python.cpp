#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tuatara.h"

namespace py = pybind11;

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

py::dict output_item_to_dict(const OutputItem &item) {
  py::dict d;
  d["text"] = item.text;
  d["bbox"] = item.bbox;
  return d;
}

py::list image_to_data_wrapper(py::array_t<unsigned char> image_data, std::string weights_dir, std::string output_dir) {
  cv::Mat img = buffer_to_mat(image_data);
  std::vector<OutputItem> items = image_to_data(img, weights_dir, output_dir);
  py::list result;
  for (const auto &item : items) {
    result.append(output_item_to_dict(item));
  }

  return result;
}

PYBIND11_MODULE(pytuatara, m) {
  m.doc() = "Tuatara ocr";

  m.def("image_to_data", &image_to_data_wrapper, py::arg("image"), py::arg("weights_dir"), py::arg("outputs_dir"), "Extract text and bounding boxes from an image");
}
