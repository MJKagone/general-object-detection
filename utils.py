import random
import cv2 as cv

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

class_colors = {}

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

def get_imgsz(dims):
    """Given the dimensions of the input frame, determine an appropriate 'imgsz' for YOLO inference."""

    max_dim = max(dims)
    if max_dim <= 640:
        return 640
    elif max_dim <= 1280:
        return 1280
    elif max_dim <= 1920:
        return 1920
    else:
        return 2560
    
def draw_bounding_boxes(frame, model, results, box_thickness=2, text_scale=0.5, text_thickness=1):
    """
    Draws bounding boxes and labels on the frame based on the model's results.
    
    Parameters:
    - frame: The OpenCV image/frame to draw on.
    - model: The model used for inference.
    - results: The results from the model inference.
    - box_thickness: Thickness of the bounding box lines.
    - text_scale: Scale factor for the label text.
    - text_thickness: Thickness of the label text.
    """
    
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = float(box.conf[0])                
            class_id = int(box.cls[0])

            # Get a color for the class
            if class_id not in class_colors:
                class_colors[class_id] = get_class_color(class_id)                
            color = class_colors[class_id]
            
            # Draw the bounding box
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw the label
            class_name = model.names[class_id]
            label_text = f"{class_name} {confidence:.2f}"
            cv.rectangle(frame, (int(x1)-1, int(y1)-20), (int(x2)+1, int(y1)), color, -1)
            cv.putText(frame, label_text, (int(x1)+1, int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (5, 5, 5), 2)