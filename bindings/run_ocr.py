import sys

sys.path.append("../build/bindings/")
import pytuatara


def main():
    pytuatara.run_ocr(
        "../images/resume_example.png",
        "../weights",
        "../outputs",
        "0"
    )


main()
