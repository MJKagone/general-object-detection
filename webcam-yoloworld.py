import os
import random
import time
import cv2 as cv
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(["eye", "mouth", "nose", "ear", "teeth", "finger", ""])
model.conf = 0.25
camera = cv.VideoCapture(0)
dim = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))
fps = camera.get(cv.CAP_PROP_FPS)
# camera.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
# camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
output = cv.VideoWriter("data/webcam_output.mp4", cv.VideoWriter_fourcc(*"avc1"), fps=15, frameSize=dim)

class_colors = {}
t_previous = 0
save = False

if __name__ == "__main__":
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        
        # Get width x height

        results = model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf[0])                
                class_id = int(box.cls[0])

                # Get the class color or assign a random one
                if class_id not in class_colors:
                    class_colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))                
                color = class_colors[class_id]
                
                # Draw the bounding box
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Get the class label
                class_name = model.names[class_id]
                label_text = f"{class_name} {confidence:.2f}"

                # Draw the label background
                cv.rectangle(frame, (int(x1)-1, int(y1)-20), (int(x2)+1, int(y1)), color, -1)
                
                # Put the text slightly above the top-left corner of the bounding box
                cv.putText(frame, label_text, (int(x1)+1, int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (5, 5, 5), 2)

        # Dynamic FPS display
        t_current = time.time()
        fps = 1 / (t_current - t_previous)
        t_previous = t_current

        fps_text = f"FPS: {int(fps)}"
        cv.putText(frame, fps_text, (dim[0]-130, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv.imshow(f"Webcam feed", frame)
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