# # tracking.py

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from resnet_extractor import FeatureExtractor
# from deep_sort.tracker import Tracker
# from deep_sort.nn_matching import NearestNeighborDistanceMetric
# from deep_sort.detection import Detection
# # === Configuration ===
# VIDEO_PATH = '15sec_input_720p.mp4'
# MODEL_PATH = 'best.pt'
# OUTPUT_PATH = 'output_tracked.avi'

# # === Initialize Models ===
# yolo_model = YOLO(MODEL_PATH)
# class_names = yolo_model.names
# extractor = FeatureExtractor()

# metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
# tracker = Tracker(metric)

# # === Video Setup ===
# cap = cv2.VideoCapture(VIDEO_PATH)
# w, h = int(cap.get(3)), int(cap.get(4))
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

# print("[INFO] Starting tracking...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # === Run YOLOv11 Detection ===
#     results = yolo_model(frame, verbose=False)[0]
#     boxes = results.boxes.xyxy.cpu().numpy()
#     scores = results.boxes.conf.cpu().numpy()
#     classes = results.boxes.cls.cpu().numpy()

#     features = []
#     detections = []

#     for box, score, cls in zip(boxes, scores, classes):
#         if class_names[int(cls)] != "player":
#             continue

#         x1, y1, x2, y2 = map(int, box)
#         crop = frame[y1:y2, x1:x2]

#         if crop.size == 0:
#             continue

#         # === Appearance Feature ===
#         feature = extractor.extract(crop)
#         features.append(feature)

#         detections.append(Detection([x1, y1, x2 - x1, y2 - y1], score, feature))

#     # === DeepSORT Update ===
#     tracker.predict()
#     tracker.update(detections)

#     # === Draw Tracks ===
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 0:
#             continue

#         x, y, w_box, h_box = map(int, track.to_tlwh())
#         track_id = track.track_id
#         cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     out.write(frame)

# # === Cleanup ===
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# print(f"[INFO] Tracking completed. Output saved as {OUTPUT_PATH}")


# tracking.py

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch
# import torchreid # Import torchreid for OSNet
# # from resnet_extractor import FeatureExtractor # No longer needed

# # DeepSORT Imports (These imports seem to be working now based on your provided code structure)
# from deep_sort.tracker import Tracker
# from deep_sort.nn_matching import NearestNeighborDistanceMetric
# from deep_sort.detection import Detection

# # === Configuration ===
# VIDEO_PATH = '15sec_input_720p.mp4'
# MODEL_PATH = 'best.pt'
# OUTPUT_PATH = 'output_tracked.avi'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === Initialize YOLOv11 ===
# yolo_model = YOLO(MODEL_PATH)
# class_names = yolo_model.names

# # === Initialize OSNet Feature Extractor (Replaces ResNet50) ===
# print("[INFO] Initializing OSNet feature extractor...")
# # Load the OSNet model, pre-trained on ImageNet and Market1501 (good for person Re-ID)
# osnet_extractor = torchreid.models.osnet_x1_0(pretrained=True).to(DEVICE)
# osnet_extractor.eval()

# # Helper function to extract features using OSNet
# def extract_features_osnet(image):
#     # OSNet expects input images normalized and resized (usually 256x128 or similar)
#     # We rely on torchreid's transforms if available, otherwise we manually resize/normalize
    
#     # Simple preprocessing: resize and convert to tensor
#     img_tensor = cv2.resize(image, (128, 256)) # Typical size for person Re-ID models
#     img_tensor = img_tensor.transpose(2, 0, 1) # HWC to CHW
#     img_tensor = np.expand_dims(img_tensor, axis=0) # Add batch dimension
#     img_tensor = torch.from_numpy(img_tensor).float() / 255.0 # Convert to tensor and normalize 
    
#     # If the image uses BGR (OpenCV default), you might need to convert to RGB here depending on OSNet's training
#     # For simplicity, we assume OSNet handles the input format or is fine with BGR for now.
    
#     img_tensor = img_tensor.to(DEVICE)
    
#     with torch.no_grad():
#         features = osnet_extractor(img_tensor).cpu().numpy()
    
#     # We return the features squeezed (remove batch dimension)
#     return features.squeeze()

# # === DeepSORT Initialization ===
# # Adjusting metric parameters based on OSNet features (0.2 from your original code is good)
# metric = NearestNeighborDistanceMetric("cosine", 0.3, 100)
# tracker = Tracker(metric, max_age=150) 

# # === Video Setup ===
# cap = cv2.VideoCapture(VIDEO_PATH)
# w, h = int(cap.get(3)), int(cap.get(4))
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

# print("[INFO] Starting tracking...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # === Run YOLOv11 Detection ===
#     results = yolo_model(frame, verbose=False)[0]
#     boxes = results.boxes.xyxy.cpu().numpy()
#     scores = results.boxes.conf.cpu().numpy()
#     classes = results.boxes.cls.cpu().numpy()

#     features = []
#     detections = []

#     for box, score, cls in zip(boxes, scores, classes):
#         if class_names[int(cls)] != "player":
#             continue

#         # Convert box to integer coordinates
#         x1, y1, x2, y2 = map(int, box)
        
#         # Crop the player image
#         crop = frame[y1:y2, x1:x2]

#         if crop.size == 0 or (x2 - x1) <= 0 or (y2 - y1) <= 0:
#             continue

#         # === Appearance Feature (Using OSNet) ===
#         feature = extract_features_osnet(crop)
#         features.append(feature)

#         # DeepSORT requires (x, y, w, h) format for bounding boxes, score, and feature
#         bbox_tlwh = [x1, y1, x2 - x1, y2 - y1]
#         detections.append(Detection(bbox_tlwh, score, feature))

#     # === DeepSORT Update ===
#     tracker.predict()
#     tracker.update(detections)

#     # === Draw Tracks ===
#     for track in tracker.tracks:
#         # Only draw confirmed tracks that are currently active
#         if not track.is_confirmed() or track.time_since_update > 1: # Increased threshold for active tracks
#             continue

#         # Get bounding box in (x, y, w, h) format
#         x, y, w_box, h_box = map(int, track.to_tlwh())
#         track_id = track.track_id
        
#         cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     out.write(frame)

# # === Cleanup ===
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# print(f"[INFO] Tracking completed. Output saved as {OUTPUT_PATH}")

import cv2
import torch
from ultralytics import YOLO
import os

# --- Configuration ---
MODEL_PATH = 'best.pt'
VIDEO_PATH = '15sec_input_720p.mp4' 
OUTPUT_VIDEO_PATH = 'output_tracked.avi' # Output video path
TRACKED_CLASSES = ['player']

# --- Initialize YOLO Model ---
print("Loading YOLOv11 model...")
# Initialize the YOLO model
model = YOLO(MODEL_PATH)
class_names = model.names
print(f"Model class names: {class_names}")

# --- Video Processing Setup ---

print("Starting tracking...")

# Get the class index for 'player'
player_class_indices = [idx for idx, name in class_names.items() if name == 'player']

# Initialize VideoWriter (OpenCV is used for saving the output)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Use 'MJPG' codec for AVI files for wide compatibility
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("Error: VideoWriter initialization failed. Cannot save output video.")
    cap.release()
    exit()

# --- Main Tracking Loop (Using model.track) ---

# We process the video using ultralytics built-in tracking capabilities.
# Since 'supervision' (ByteTrack dependencies) is installed, ultralytics should use a stable tracker.

results = model.track(source=VIDEO_PATH, 
                      conf=0.5,           # Detection confidence threshold
                      persist=True,       # Maintain tracks across frames
                      verbose=False,      
                      stream=True,        # Process the video stream
                      save=False)         # We save frames manually below

# Process the results and visualize
for frame_idx, result in enumerate(results):
    frame = result.orig_img
    
    # Check if tracking results are present (i.e., if track_id exists)
    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
        
        # Extract tracked data
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # Iterate through tracked objects
        for box, track_id, conf, class_id in zip(boxes, track_ids, confidences, class_ids):
            
            # Filter for 'player' class
            if player_class_indices and int(class_id) not in player_class_indices:
                continue

            # Extract coordinates and track ID
            x1, y1, x2, y2 = map(int, box)
            
            # Define the color and label for the tracked object
            color = (0, 255, 0) # Green
            label = f'ID: {int(track_id)} ({conf:.2f})'

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display the tracking ID
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Write the frame to the output video
    out.write(frame)

# --- Cleanup ---
out.release()
cv2.destroyAllWindows()
print(f"Tracking finished. Output video saved to {OUTPUT_VIDEO_PATH}")