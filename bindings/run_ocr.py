import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append("../build/bindings/")
import pytuatara


def main():
    # pytuatara.image_to_data(
    #     "../images/resume_example.png",
    #     "../weights",
    #     "../outputs",
    #     "0"
    # )


    image = Image.open('/Users/jackvial/Code/CPlusPlus/tuatara/images/art-01107.jpg')
    numpy_image = np.array(image)
    sobel_image_numpy = pytuatara.image_to_data(numpy_image, "../weights", "../outputs", "0")
    
    # Convert the filtered NumPy array back to a PIL image
    sobel_image = Image.fromarray(sobel_image_numpy)

    # Save the filtered image
    sobel_image.save('/Users/jackvial/Code/CPlusPlus/tuatara/images/art-01107-interface-test.jpg')

main()
