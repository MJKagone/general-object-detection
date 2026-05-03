import argparse
import base64
from openai import OpenAI
import cv2 as cv
from utils import overlay_grid

def main(prompt, encoded_image_bytes):

    client = OpenAI()
    
    # 1. Convert the raw bytes from OpenCV to a Base64 string
    base64_image = base64.b64encode(encoded_image_bytes).decode('utf-8')

    # 2. Use the new Responses API endpoint
    response = client.responses.create(
        model="gpt-5.4", 
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", #[cite: 1]
                        "text": f"Output the grid coordinates containing the following object in this image: <<{prompt}>>. Use JSON format with the following schema: {{'coordinates': [[A1, A2], [C3, C4, D3, D4], [B2]]}} (in this example there are three instances of the object, one in the horizontal region defined by A1 and A2, one in the square defined by C3, D3, C4, D4, and one in the single grid cell B2). If the object is not found, output {{'coordinates': []}}"
                    },
                    {
                        "type": "input_image", #[cite: 1]
                        # 3. Format as a Data URI[cite: 1]
                        "image_url": f"data:image/jpeg;base64,{base64_image}" 
                    }
                ]
            }
        ]
    )

    # 4. Access the response using the updated attribute[cite: 1]
    print(response.output_text) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for OpenAI API")
    parser.add_argument("-i", "--image", type=str, help="Path to the local image file to be processed")
    parser.add_argument("-p", "--prompt", type=str, help="The object to find in the image")
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        print("Error: Could not read image.")
        exit(1)

    image_grid = overlay_grid(img, 5*5)
    
    cv.imwrite("data/image_grid.jpg", image_grid)
    success, img_encoded = cv.imencode(".jpg", image_grid)
    
    if success:
        img_bytes = img_encoded.tobytes()
        main(args.prompt, img_bytes)
    else:
        print("Error encoding image")