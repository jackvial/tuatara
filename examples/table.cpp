#include "tuatara.h"

// Main file
int main(int argc, const char** argv) {
    image_to_data(
        "../../images/table_english.png",
        "../../weights",
        "../../outputs",
        "0"
    );
    return 0;
}