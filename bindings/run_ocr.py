import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append("../build/bindings/")
import pytuatara


def main():
    image = Image.open('/Users/jackvial/Code/CPlusPlus/tuatara/images/resume_example.png')
    numpy_image = np.array(image)
    result = pytuatara.image_to_data(numpy_image, "../weights", "../outputs", "0")
    print(result)

main()
