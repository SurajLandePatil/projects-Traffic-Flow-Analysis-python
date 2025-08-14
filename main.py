import cv2
import numpy as np
import torch
import pandas as pd
from sort import Sort
from datetime import timedelta

cap = cv2.VideoCapture("traffic.mp4")  # Your local video

# YOLOv5 setup
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
vehicle_classes = [2, 3, 5, 7]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# ----------- Define Lane boundaries and green line (edit these numbers using image inspection) -----------
# Example lane x boundaries (edit for your video):
lane_boundaries = [
    int(frame_width*0.40),   # Lane 1-2 boundary (based on your image)
    int(frame_width*0.60),   # Lane 2-3 boundary
    int(frame_width*0.90)    # Lane 3 end (for visual, not used in calculation)
]

# The green line: define its y-position and span across the road (edit as per image, e.g. bottom 1/5 of ROI)
green_line_y = 480  # example: if video height is 720, and green line is at y=480, edit as necessary

# Lane regions: spans between lane boundary x positions
lane_regions = [
    (0, lane_boundaries[0]),          # Lane 1: from left edge to first boundary
    (lane_boundaries[0], lane_boundaries[1]),  # Lane 2
    (lane_boundaries[1], lane_boundaries[2])   # Lane 3: from boundary to right edge
]

def get_lane_id(cx):
    for idx, (x1, x2) in enumerate(lane_regions):
        if x1 <= cx <= x2:
            return idx
    return None

# Counting setup
tracker = Sort()
counted_ids_per_lane = [set(), set(), set()]
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # ---- Detection ----
    results_yolo = model(frame)
    detections = []
    for *xyxy, conf, cls in results_yolo.xyxy[0]:
        if int(cls) in vehicle_classes:
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append([x1, y1, x2, y2, float(conf)])

    np_detections = np.array(detections)
    tracked_objects = tracker.update(np_detections) if len(detections) > 0 else []

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        lane_id = get_lane_id(cx)
        if lane_id is not None:
            # Count only vehicles crossing the green line in their lane band
            threshold = 12  # pixel tolerance for crossing, adjust as needed
            if green_line_y-threshold <= cy <= green_line_y+threshold:
                if int(obj_id) not in counted_ids_per_lane[lane_id]:
                    counted_ids_per_lane[lane_id].add(int(obj_id))
                    timestamp = str(timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)))
                    results.append([int(obj_id), lane_id+1, timestamp])
        # Draw bounding box, ID and lane
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        if lane_id is not None:
            cv2.putText(frame, f'ID {int(obj_id)} Lane {lane_id+1}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

    # Draw green counting line
    cv2.line(frame, (0, green_line_y), (frame_width, green_line_y), (0,255,0), 3)
    cv2.putText(frame, "Counting Line", (20, green_line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Draw lane lines and counts
    for idx, x in enumerate(lane_boundaries):
        cv2.line(frame, (x, 0), (x, frame_height), (255,0,0), 2)
    for idx, (x1, x2) in enumerate(lane_regions):
        count = len(counted_ids_per_lane[idx])
        cv2.putText(frame, f"Lane {idx+1} Count: {count}", (x1+10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow('Traffic Lane Counter', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# ---- Save Results ----
df = pd.DataFrame(results, columns=["Vehicle ID", "Lane", "Timestamp"])
df.to_csv("vehicle_counts.csv", index=False)
cap.release()
cv2.destroyAllWindows()

print("Final lane counts:")
for idx, ids in enumerate(counted_ids_per_lane):
    print(f"Lane {idx+1}: {len(ids)}")
