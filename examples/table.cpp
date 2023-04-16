#include "../tuatara.h"

// Main file
int main(int argc, const char** argv) {
    run_ocr(
        "../../images/table_english.png",
        "../../weights",
        "../../outputs"
    );
    return 0;
}