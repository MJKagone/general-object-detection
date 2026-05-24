import argparse
import os
import ollama
from PIL import Image
import json
import cv2 as cv

def query_by_text(model, img_path, prompt):
    prompt_text = f"Detect all instances of the following object in the image: <<{prompt}>>. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    response = ollama.chat(model=model,
                           messages=[{
                               'role': 'user',
                               'content': prompt_text,
                               'images': [img_path]
                           }],
                           format='json')
    return response['message']['content']

def query_by_image(model, img1_path, img2_path):
    prompt_text = f"Detect all instances of the object(s) highlighted in the second image from the first image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    response = ollama.chat(model=model,
                           messages=[{
                               'role': 'user',
                               'content': prompt_text,
                               'images': [img1_path, img2_path]
                           }],
                           format='json')
    return response['message']['content']

def query_by_bbox(model, img_path):
    prompt_text = f"Is the provided bounding box optimal for the highlighted object? If not, provide a better bounding box. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    response = ollama.chat(model=model,
                           messages=[{
                               'role': 'user',
                               'content': prompt_text,
                               'images': [img_path]
                           }],
                           format='json')
    return response['message']['content']

def main(model, image_path, input_prompt):

    img = Image.open(image_path)
    
    try:
        img2 = Image.open(input_prompt)
        if input_prompt == image_path + "_bboxes.jpg":
            response_text = query_by_bbox(model, image_path)
        else:
            response_text = query_by_image(model, image_path, input_prompt)
    except FileNotFoundError:
        response_text = query_by_text(model, image_path, input_prompt)

    
    print("Raw response:", response_text)

    # Clean up markdown code blocks if the model outputs them
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    width, height = img.size
    bounding_boxes = json.loads(response_text)
    
    # Ensure bounding_boxes is a list of dictionaries
    if isinstance(bounding_boxes, dict):
        bounding_boxes = [bounding_boxes]

    converted_bounding_boxes = []
    for bounding_box in bounding_boxes:
        if "box_2d" not in bounding_box:
            continue
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
    parser = argparse.ArgumentParser(description="Example script for local Gemma via Ollama Python API")
    parser.add_argument("-i", "--image", type=str, help="Path to the local image file to be processed")
    parser.add_argument("-p", "--prompt", type=str, help="The object to find in the image")
    parser.add_argument("-m", "--model", type=str, default="gemma4", help="The Ollama model to use")
    args = parser.parse_args()

    main(args.model, args.image, args.prompt)
