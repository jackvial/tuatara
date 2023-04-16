#include "../tuatara.h"

// Main file
int main(int argc, const char** argv) {
    run_ocr(
        "../../images/resume_example.png",
        "../../weights",
        "../../outputs"
    );
    return 0;
}