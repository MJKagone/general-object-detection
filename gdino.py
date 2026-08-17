import argparse
import os
import random
import cv2 as cv
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

COLORS_RGB = [
    (255, 0, 0),     # 0: Red
    (0, 255, 0),     # 1: Green
    (0, 0, 255),     # 2: Blue
    (255, 255, 0),   # 3: Yellow
    (255, 0, 255),   # 4: Magenta
    (0, 255, 255),   # 5: Cyan
    (255, 165, 0),   # 6: Orange
    (128, 0, 128),   # 7: Purple
    (0, 128, 128),   # 8: Teal
    (128, 128, 0),   # 9: Olive
    (255, 192, 203), # 10: Pink
    (165, 42, 42),   # 11: Brown
    (0, 0, 128),     # 12: Navy
    (128, 0, 0),     # 13: Maroon
    (127, 255, 0),   # 14: Lime/Chartreuse
    (255, 127, 80),  # 15: Coral
    (255, 215, 0),   # 16: Gold
    (127, 255, 212), # 17: Aquamarine
    (218, 112, 214), # 18: Orchid
    (192, 192, 192)  # 19: Silver
]

COLORS_BGR = [(b, g, r) for r, g, b in COLORS_RGB]

def get_class_color(class_id):
    """Returns a distinct BGR color from the fixed list, or a seeded random color."""

    if class_id < len(COLORS_BGR):
        return COLORS_BGR[class_id]
    else:
        random.seed(class_id)
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        
        # Boost a channel to prevent overly dark bounding boxes
        if r < 100 and g < 100 and b < 100:
            channel = random.choice(["b", "g", "r"])
            if channel == "b": b = random.randint(150, 255)
            elif channel == "g": g = random.randint(150, 255)
            else: r = random.randint(150, 255)
            
        return (b, g, r)

# --- Setup Device ---
# This will automatically use your GPU if available, otherwise it falls back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# --- Load Model & Processor ---
# We use the base model from IDEA-Research hosted on Hugging Face
MODEL_ID = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

def draw_bounding_boxes(frame, results, classes):
    """Draws bounding boxes and labels on the OpenCV frame."""
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"] # These are string labels returned by the model

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        
        # Color coding based on class label
        color = get_class_color(hash(label) % 1000)  # Hash the label
        
        # Draw the bounding box
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw the label
        label_text = f"{label} {score:.2f}"
        (text_width, text_height), baseline = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv.putText(frame, label_text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding DINO Object Detection")
    parser.add_argument("--image", type=str, help="Path to an image file for testing, or empty for webcam")
    args = parser.parse_args()

    # --- Define Classes ---
    # Grounding DINO requires a single string separated by periods
    if not args.image:
        text_prompt = "eye. mouth. nose. ear. teeth. finger."
    elif "pc" in args.image.lower():
        text_prompt = "mass storage (PC component)."
    elif "shelf" in args.image.lower():
        text_prompt = "book. toy figure. plush toy."
    else:
        text_prompt = "object."

    if args.image:
        # --- Image Inference ---
        frame = cv.imread(args.image)
        # Convert BGR (OpenCV) to RGB (PIL) for the model
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Prepare inputs
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process (convert outputs to bounding boxes)
        # threshold handles how confident it needs to be an object
        # text_threshold handles how confident it matches your text
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[pil_image.size[::-1]]
        )

        frame = draw_bounding_boxes(frame, results, text_prompt)

        cv.namedWindow("Grounding DINO", cv.WINDOW_NORMAL)
        cv.resizeWindow("Grounding DINO", 1080, 720)
        cv.imshow("Grounding DINO", frame)
        cv.waitKey(0)

    else:
        # --- Webcam Inference ---
        camera = cv.VideoCapture(0)
        
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.25,
                text_threshold=0.25,
                target_sizes=[pil_image.size[::-1]]
            )

            frame = draw_bounding_boxes(frame, results, text_prompt)

            cv.imshow("Grounding DINO Webcam", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        camera.release()
        cv.destroyAllWindows()