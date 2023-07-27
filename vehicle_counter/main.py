import math

import cv2
import numpy as np
import sort
from classes import class_names
from ultralytics import YOLO
from utils import Colors, Point, change_to_cwd

change_to_cwd()

# Define YOLO model to use.
model = YOLO("../yolo_weights/yolov8l.pt")

# Define tracker object.
tracker = sort.Sort(max_age=20)

# Define video and mask.
video = cv2.VideoCapture("../videos/Cars.mp4")
mask = cv2.imread("../masks/Cars.png")

# Intiailize set to store counted cars.
cars_counted = set()

# Temporarily define fixed line.
start_line = Point(x=320, y=297)
end_line = Point(x=637, y=297)

while True:
    success, img = video.read()
    img_region = cv2.bitwise_and(img, mask)

    results = model(img_region, stream=True)
    detections = np.empty((0, 5))

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            confidence = math.ceil(box.conf[0] * 100) / 100
            label = class_names[int(box.cls[0])]

            if label in ("car", "truck", "bus", "motorbike") and confidence > 0.4:
                # This is for the object tracker.
                current_array = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_array))

    tracker_results = tracker.update(detections)
    cv2.line(img, start_line, end_line, Colors.RED, 4)

    for result in tracker_results:
        x1, y1, x2, y2, id_ = [int(i) for i in result]
        cv2.rectangle(img, (x1, y1), (x2, y2), Colors.BLUE, 2)
        cv2.putText(img, str(id_), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)

        center_x, center_y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (center_x, center_y), 4, Colors.BLUE, cv2.FILLED)

        if start_line.x < center_x < end_line.x and start_line.y - 10 < center_y < start_line.y + 10 and id_ not in cars_counted:
            cars_counted.add(id_)

        cv2.putText(img, f"Cars: {len(cars_counted)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, Colors.GREEN, 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
