import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append("../build/bindings/")
import pytuatara

def draw_boxes_and_text(image, results):
    for result in results:
        text = result['text']
        bbox = result['bbox']
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw bounding box
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        # Put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_x = x
        text_y = y - 5
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    return image

def main():
    
    # @TODO  - update the interface so it can take a PIL Image object and apply .convert("RGB") internally
    # so the user doesn't have to worry about it. Also allow for passing a raw numpy array.
    image = Image.open('/Users/jackvial/Code/CPlusPlus/tuatara/images/resume_example.png').convert("RGB")
    numpy_image = np.array(image)
    result = pytuatara.image_to_data(numpy_image, "../weights", "../outputs")
    print(result)
    
    annotated_image = draw_boxes_and_text(numpy_image, result)

    # Display the annotated image in a window
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
