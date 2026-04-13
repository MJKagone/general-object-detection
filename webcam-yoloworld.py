import argparse
import os
import random
import time
import cv2 as cv
from ultralytics import YOLOWorld

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
    
def draw_bounding_boxes(frame, results, box_thickness=2, text_scale=0.5, text_thickness=1):
    """Draws bounding boxes and labels on the frame based on YOLO results."""
    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Webcam YOLOWorld Object Detection")
    parser.add_argument("--image", type=str, help="Path to an image file for testing, or empty for webcam")
    args = parser.parse_args()

    model = YOLOWorld("models/yolov8x-worldv2.pt")
    if not args.image:
        model.set_classes(["eye", "mouth", "nose", "ear", "teeth", "finger", ""])
    elif "pc" in args.image.lower():
        model.set_classes(["cooling fan", "GPU", "electrical cables", "motherboard", ""])
    elif "shelf" in args.image.lower():
        model.set_classes(["book", "toy figure", "plush toy", ""])
    camera = cv.VideoCapture(0)
    dim = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))
    fps = camera.get(cv.CAP_PROP_FPS)
    # camera.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    # camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    output = cv.VideoWriter("data/webcam_output.mp4", cv.VideoWriter_fourcc(*"avc1"), fps=15, frameSize=dim)

    class_colors = {}
    t_previous = 0
    save = False

    if args.image:

        img = cv.imread(args.image)

        # Create a resizeable window
        cv.namedWindow("OpenCV", cv.WINDOW_NORMAL)
        cv.resizeWindow("OpenCV", 1080, 720)

        results = model(img, imgsz=get_imgsz(img.shape[:2]), conf=0.05, iou=0.25, verbose=False)
        # Calculate a dynamic scale factor based on the image width
        # (Assuming a 1080px wide image looks good with default settings)
        scale_ratio = img.shape[1] / 1080.0
        
        # Set dynamic thickness and font size based on the ratio
        box_thickness = max(2, int(2 * scale_ratio))
        text_scale = max(0.5, 0.5 * scale_ratio)
        text_thickness = max(1, int(2 * scale_ratio))

        draw_bounding_boxes(img, results, box_thickness=box_thickness, text_scale=text_scale, text_thickness=text_thickness)

        # Show the high-res image inside the resized window
        cv.imshow("OpenCV", img)
        cv.waitKey(0)

    else:

        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            
            results = model(frame, conf=0.1, iou=0.25, imgsz=get_imgsz(frame.shape[:2]), verbose=False, stream=True)
            draw_bounding_boxes(frame, results)

            # Dynamic FPS display
            t_current = time.time()
            fps = 1 / (t_current - t_previous)
            t_previous = t_current

            fps_text = f"FPS: {int(fps)}"
            cv.putText(frame, fps_text, (dim[0]-130, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv.imshow(f"OpenCV", frame)
            output.write(frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save = True
                break


    # Clean up resources
    camera.release()
    output.release()
    cv.destroyAllWindows()

    if not save:
        os.remove("data/webcam_output.mp4")