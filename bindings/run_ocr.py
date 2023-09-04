import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append("../build/bindings/")
import pytuatara


def draw_boxes_and_text(image, results):
    # Sort the results based on the y and x coordinates
    sorted_results = sorted(results, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Create blank images of the same size as the input image
    h, w, c = image.shape
    blank_image_with_text = np.zeros((h, w, c), dtype=np.uint8)
    blank_image_sorted_text = np.zeros((h, w, c), dtype=np.uint8)

    # Initialize variables for the sorted text image
    sorted_text_x = 10
    sorted_text_y = 30
    font_sorted = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_sorted = 0.5
    font_thickness_sorted = 1
    line_spacing = 10

    for result in sorted_results:
        text = result["text"]
        bbox = result["bbox"]
        x, y, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        box_width, box_height = x2 - x, y2 - y

        # Draw the rectangle on the original image
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

        # Scale the text to fit inside the box with an upper limit
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, 1, font_thickness)
        text_width, text_height = text_size
        font_scale = min(box_width / text_width, box_height / text_height, 1)

        # Draw the text on both the original image and the blank image with text inside the boxes
        text_x = x
        text_y = y + int(text_height * font_scale)
        cv2.putText(
            blank_image_with_text,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )

        # Draw the sorted text on the blank image
        sorted_text_size, _ = cv2.getTextSize(
            text, font_sorted, font_scale_sorted, font_thickness_sorted
        )
        sorted_text_width, sorted_text_height = sorted_text_size

        if sorted_text_x + sorted_text_width > w:
            sorted_text_x = 10
            sorted_text_y += sorted_text_height + line_spacing

        cv2.putText(
            blank_image_sorted_text,
            text,
            (sorted_text_x, sorted_text_y),
            font_sorted,
            font_scale_sorted,
            (0, 0, 255),
            font_thickness_sorted,
        )
        sorted_text_x += sorted_text_width + line_spacing

    # Concatenate the images horizontally
    combined_image = cv2.hconcat(
        [image, blank_image_with_text, blank_image_sorted_text]
    )

    return combined_image


def main():
    # @TODO  - update the interface so it can take a PIL Image object and apply .convert("RGB") internally
    # so the user doesn't have to worry about it. Also allow for passing a raw numpy array.
    image = Image.open(
        "../images/funsd_0001129658.png"
    ).convert("RGB")
    numpy_image = np.array(image)
    result = pytuatara.image_to_data(numpy_image, "../../models", "../outputs")
    print(result)

    annotated_image = draw_boxes_and_text(numpy_image, result)
    
    # Save the annotated image
    cv2.imwrite("../outputs/funsd_0001129658_annotated_with_ocr_results.png", annotated_image)

    # Display the annotated image in a window
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


main()
