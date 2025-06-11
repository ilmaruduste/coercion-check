import os
import cv2
import numpy as np
import pandas as pd
import torch
import glob
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_coercion_check(video_paths, model=None, conf_threshold=0.6, save_visualizations=True, output_path=None):
    """
    Main function to check coercion in videos by detecting and tracking multiple people.
    
    Args:
        video_paths (list): List of video file paths to analyze
        model (YOLO, optional): Pre-loaded YOLO model, or None to load from config
        conf_threshold (float, optional): Confidence threshold for person detection
        save_visualizations (bool): Whether to save visualizations to log directory
    
    Returns:
        pd.DataFrame: Results dataframe with detected persons and coercion classification
    """
    # Initialize model if not provided
    if model is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model = YOLO(os.path.join("models", "yolo_model8m.pt")).to(device)

    # Ensure log directory exists
    os.makedirs(output_path, exist_ok=True)
    
    coercion_results = []
    tracker = DeepSort(max_age=30)
    
    for video_path in video_paths:
        print(f"\nProcessing {video_path}...")
        results = person_detection_and_tracking(video_path, model, tracker, conf_threshold, save_visualizations=save_visualizations, output_path=output_path)

        # Determine if coercion is detected (multiple people)
        coercion_detected = int(len(results) > 1)
        video_name = os.path.basename(video_path).split(".")[0]
        
        print(f"Found {len(results)} unique persons in {video_name}")
        
        # Create results entry for each detected person
        for person_id, data in results.items():
            frames = data["frames"]
            confidences = data["confidences"]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            coercion_results.append({
                "video_path": video_path,
                "video": video_name,
                "person_id": person_id,
                "num_frames": len(frames),
                "avg_confidence": avg_conf,
                "confidence_threshold": conf_threshold,
                "first_frame": frames[0] if frames else 0,
                "last_frame": frames[-1] if frames else 0,
                "coercion_detected": coercion_detected
            })

    coercion_results_df = pd.DataFrame(coercion_results)
    print(f"Processed {coercion_results_df.video.nunique()} unique videos with coercion detection results.")
    
    return coercion_results_df

def person_detection_and_tracking(video_path, model, tracker, conf_threshold=0.6, save_visualizations=True, output_path=None):
    """
    Detects and tracks persons in a video using YOLO and DeepSORT.
    
    Args:
        video_path (str): Path to the video file
        model (YOLO): YOLO model for person detection
        tracker (DeepSort): Deep SORT tracker
        conf_threshold (float): Confidence threshold for detections
        save_visualizations (bool): Whether to save visualizations to output directory

    Returns:
        dict: Dictionary with information about tracked persons
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Setup output video if requested
    if save_visualizations:
        video_output_dir = os.path.join(output_path, "processed_videos")
        os.makedirs(video_output_dir, exist_ok=True)
        
        video_name = os.path.basename(video_path).split(".")[0]
        output_path = os.path.join(video_output_dir, f"{video_name}_tracked.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(3)), int(cap.get(4))))

    # Track persons across frames
    person_items = defaultdict(lambda: {"frames": [], "confidences": []})
    frame_idx = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections using YOLO with confidence threshold
        results = model(frame, verbose=False, conf=conf_threshold)[0]

        # Format detections for Deep SORT
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:  # person class only
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 'person'])

        # Update Deep SORT tracks
        tracks = tracker.update_tracks(detections, frame=frame)

        # Process each confirmed track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Find matching detection and its confidence
            detection_conf = 0
            for det, conf, _ in detections:
                det_x1, det_y1, det_w, det_h = det
                det_x2, det_y2 = det_x1 + det_w, det_y1 + det_h
                # Check if detection box matches track box
                if abs(l - det_x1) < 20 and abs(t - det_y1) < 20 and abs(r - det_x2) < 20 and abs(b - det_y2) < 20:
                    detection_conf = conf
                    break

            # Log the frame and confidence for this person
            person_items[track_id]["frames"].append(frame_idx)
            person_items[track_id]["confidences"].append(detection_conf)

            # Draw box and label if saving output
            if save_visualizations:
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                label = f"Person {track_id} ({detection_conf:.2f})"
                cv2.putText(frame, label, (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write the frame to output video if requested
        if save_visualizations:
            out.write(frame)
            
        frame_idx += 1

    # Release resources
    cap.release()
    if save_visualizations:
        out.release()

    return person_items