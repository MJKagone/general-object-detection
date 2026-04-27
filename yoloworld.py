import argparse
import os
import time

import cv2 as cv
from ultralytics import YOLOWorld

from utils import *

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

        draw_bounding_boxes(img, model, results, box_thickness=box_thickness, text_scale=text_scale, text_thickness=text_thickness)

        # Show the high-res image inside the resized window
        cv.imshow("OpenCV", img)
        cv.waitKey(0)

    else:

        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            
            results = model(frame, conf=0.1, iou=0.25, imgsz=get_imgsz(frame.shape[:2]), verbose=False, stream=True)
            draw_bounding_boxes(frame, model, results)

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