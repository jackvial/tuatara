#include <pybind11/pybind11.h>
#include "tuatara.h"

namespace py = pybind11;

PYBIND11_MODULE(pytuatara, m) {
    m.doc() = "Tuatara ocr";

    m.def("run_ocr", &run_ocr, "Runs OCR on given image path and saves to given output path");
}
