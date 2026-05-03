import argparse
import os
from google import genai
from google.genai import types
from PIL import Image
import json
import cv2 as cv

def main(image_path, prompt):

    img = Image.open(image_path)

    client = genai.Client()
    prompt = f"Detect all instances of the following object in the image: <<{prompt}>>. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

    config = types.GenerateContentConfig(
    response_mime_type="application/json"
    )

    response = client.models.generate_content(model="gemini-3-flash-preview",
                                            contents=[img, prompt],
                                            config=config
                                            )

    width, height = img.size
    bounding_boxes = json.loads(response.text)

    converted_bounding_boxes = []
    for bounding_box in bounding_boxes:
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

    # print("Image size: ", width, height)
    # print("Bounding boxes:", converted_bounding_boxes)

    img = cv.imread(image_path)
    for box in converted_bounding_boxes:
        cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    output_path = os.path.join("data/output", f"{name}_bboxes.jpg")
    os.makedirs("data/output", exist_ok=True)
    
    cv.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for Google Gemini API")
    parser.add_argument("-i", "--image", type=str, help="Path to the local image file to be processed")
    parser.add_argument("-p", "--prompt", type=str, help="The object to find in the image")
    args = parser.parse_args()

    main(args.image, args.prompt)
