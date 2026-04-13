import argparse
import math
import numpy as np
from ollama import chat
import cv2 as cv

def overlay_grid(img, num_patches):
    """
    Takes an image, divides it into patches and overlays a grid on the original image to show the patch boundaries.

    Args:
        img: The image to be divided into patches. Can be a file path or a numpy image array.
        num_patches: The number of patches to be created. Must be a perfect square.
        
    Returns:
        A copy of the original image with rectangles drawn around the patches.
    """

    if not math.sqrt(num_patches).is_integer():
        raise ValueError("num_patches must be a perfect square")
    
    if isinstance(img, str):
        img = cv.imread(img)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img}")
    elif isinstance(img, np.ndarray):
        img = img
    else:
        raise TypeError("img must be a file path or a numpy image array")
    
    h, w, _ = img.shape

    offset = int(math.sqrt(num_patches))
    offset_w = int(w / offset)
    offset_h = int(h / offset)

    padding = 40
    # Add padding to the top and left for labels
    img_copy = cv.copyMakeBorder(img, padding, 0, padding, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])

    for row in range(offset):
        # Draw row label (A, B, C...)
        text_y = padding + row * offset_h + offset_h // 2
        cv.putText(img_copy, chr(65 + row), (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for col in range(offset):
            if row == 0:
                # Draw col label (1, 2, 3...)
                text_x = padding + col * offset_w + offset_w // 2 - 10
                cv.putText(img_copy, str(col + 1), (text_x, padding - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            start_w = padding + col * offset_w
            start_h = padding + row * offset_h
            end_w = start_w + offset_w
            end_h = start_h + offset_h

            cv.rectangle(img_copy, (start_w, start_h), (end_w-1, end_h-1), (0, 0, 255), 2)

    return img_copy

def get_region_proposals(image_bytes, to_detect):
    """
    Uses a vision language model to analyze the image and propose regions of interest based on the object we want to detect.
    
    Args:
        image_bytes: The image data in bytes format.
        to_detect: A string describing the object we want to detect (e.g., 'dog').
    
    Returns:
        A list of proposed regions in grid coordinates (e.g., ["A1", "C2, C3, D2, D3"]).
    """

    print("🤖 Analyzing image with VLM...")

    response = chat(
        model="gemma4",
        messages= [
                    {
                        "role": "user",
                        "content": f"""Look at this image and find regions that look like they might contain the object of interest: '{to_detect}'. The regions will be forwarded to a downstream model for further analysis.
                        
                        1. First, briefly describe the areas in the image that look like they might contain an instance of the object and where they are located.
                        2. Then, map that location to the overlayed grid.
                        3. Finally, provide the exact grid cells as a JSON list of lists. The regions can be individual cells (e.g., "A1") or groups of adjacent cells (e.g., "C2, C3, D2, D3").
                        
                        Example final output format:
                        [
                            {{"Coordinates": [("A1"), ("C2", "C3", "D2", "D3")]}}
                        ]
                        """,
                        "images": [image_bytes]
                    }
                ]
    )
    return response.message.content.strip()

# CLI usage: python query.py [url] (defaults to vision model, which seems to perform better)
# CLI usage: python query.py -m both [url] (to see both vision and text predictions)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--to_detect", type=str, help="The object to detect in the image")
    parser.add_argument("--image", type=str, help="Path to an image file")
    args = parser.parse_args()
    img_bytes = None
        
    img = cv.imread(args.image)
    img_grid = overlay_grid(img, 5*5)
    cv.imwrite("data/image_grid.jpg", img_grid)
    _, img_encoded = cv.imencode(".jpg", img_grid)
    img_bytes = img_encoded.tobytes()
    if img_bytes:
        regions = get_region_proposals(img_bytes, args.to_detect)
        print(regions)
    else:
        print("Failed to read image or convert to bytes.")