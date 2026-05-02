import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
from collections import deque
from ultralytics import YOLO

# Ensure keras compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from model_config import MODELS_CONFIG
from benchmark import load_spil_model, load_stgcn_model, load_poseconv3d_model

# Import preprocessing logic extracted from generators
from spil import get_features_spil_from_yolo_results
from stgcn import get_features_graph_from_yolo_results, build_graph, separate_features_and_label
from train import get_features_conv3d_from_yolo_results, limb_heatmap, format_frames
from preprocessing import STEP, HEIGHT, WIDTH, CHANNELS

def pad_or_sample_points(points, n_points):
    """Pad or sample points to reach exactly n_points (Used for SPIL)."""
    if len(points) == 0:
        return np.zeros((n_points, 4), dtype=np.float32)
    
    if len(points) < n_points:
        indices = np.random.choice(len(points), n_points, replace=True)
    else:
        indices = np.random.choice(len(points), n_points, replace=False)
    
    return np.array(points)[indices].astype(np.float32)


def process_video(input_path, output_path, config):
    model_type = config["type"]
    n_frames = config.get("n_frames", 10)
    frame_step = STEP # From preprocessing.py
    name = config["name"]
    
    print(f"Loading YOLO model...")
    yolo_model = YOLO("yolo11n-pose.pt")
    
    print(f"Loading {name} model...")
    if "SPIL" in model_type:
        model, _ = load_spil_model(config)
        n_points = config.get("n_points", 1024)
    elif "ST_GCN" in model_type:
        model, _ = load_stgcn_model(config)
    elif "PoseConv3D" in model_type:
        model, _ = load_poseconv3d_model(config)
    else:
        print(f"Unsupported model type: {model_type}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buffer_size = (n_frames - 1) * frame_step + 1
    yolo_buffer = deque(maxlen=buffer_size) 
    shape_buffer = deque(maxlen=buffer_size)
    
    print(f"Processing video with {model_type}...")
    print(f"Window Size: {buffer_size} frames (sampling {n_frames} frames with step {frame_step})")
    
    frame_count = 0
    current_label = "Buffering..."
    current_color = (0, 255, 255) # Yellow in BGR
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Feature Extraction with YOLO
        results = yolo_model(frame, verbose=False)
        yolo_buffer.append(results)
        shape_buffer.append(frame.shape)
        
        # 2. Run Inference when total frames are collected
        if len(yolo_buffer) == buffer_size:
            # Subsample the buffers to match the training data spacing
            sampled_results = list(yolo_buffer)[::frame_step]
            sampled_shapes = list(shape_buffer)[::frame_step]
            
            preds = None
            
            # Use exact same preprocessing  generators
            if "SPIL" in model_type:
                points = get_features_spil_from_yolo_results(sampled_results)
                sampled_points = pad_or_sample_points(points, n_points)
                input_data = np.expand_dims(sampled_points, axis=0) # Shape: (1, n_points, 4)
                preds = model.predict(input_data, verbose=0)[0] 
                
            elif "ST_GCN" in model_type:
                joints_all, limbs_all, joints_time = get_features_graph_from_yolo_results(sampled_results)
                # Pass 0 as dummy label
                graph = build_graph(joints_all, limbs_all, joints_time, [0]) 
                # Create a batch of size 1
                ds = tf.data.Dataset.from_tensors(graph).batch(1)
                ds = ds.map(separate_features_and_label)
                for features, _ in ds:
                    preds = model.predict(features, verbose=0)[0]
                    break
                    
            elif "PoseConv3D" in model_type:
                formatted_frames = []
                for i in range(n_frames):
                    feats = get_features_conv3d_from_yolo_results(sampled_results[i], sampled_shapes[i], limb_heatmap, (HEIGHT, WIDTH, CHANNELS))
                    formatted_frame = format_frames(feats, (HEIGHT, WIDTH))
                    formatted_frames.append(formatted_frame)
                input_data = np.expand_dims(np.array(formatted_frames), axis=0) # Shape: (1, n_frames, H, W, C)
                preds = model.predict(input_data, verbose=0)[0]
            
            if preds is not None:
                pred_class = np.argmax(preds)
                prob = preds[pred_class]
                
                # Labels: 0 for Violence, 1 for NonViolence (from preprocessing.py get_class_ids)
                if pred_class == 0:
                    current_label = f"Violence: {prob:.2f}"
                    current_color = (0, 0, 255) # Red 
                else:
                    current_label = f"Non-Violence: {prob:.2f}"
                    current_color = (0, 255, 0) # Green 

        # 3. Draw Annotations
        annotated_frame = results[0].plot() 
        
        # Add a black background rectangle for the text to be visible
        (text_width, text_height), baseline = cv2.getTextSize(current_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(annotated_frame, (40, 20), (50 + text_width, 60 + text_height), (0, 0, 0), -1)
        cv2.putText(annotated_frame, current_label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2, cv2.LINE_AA)
        
        out.write(annotated_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Finished processing. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Video Inference")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--model", type=str, default="SPIL-f10", help="Model name from config (e.g. SPIL-f10)")
    
    args = parser.parse_args()
    
    config = next((m for m in MODELS_CONFIG if m["name"] == args.model), None)
    if not config:
        print(f"Error: Model '{args.model}' not found in model_config.py")
        print("Available models:")
        for m in MODELS_CONFIG:
            print(f" - {m['name']}")
        exit(1)
        
    process_video(args.input, args.output, config)
