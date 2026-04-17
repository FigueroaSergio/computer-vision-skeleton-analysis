import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import csv
from model_config import MODELS_CONFIG


# Import existing components
try:
    import stgcn
    import spil
    from PoseCon3d import Pose3D
    from preprocessing import get_frames, get_class_ids
    from train import get_dataset, FrameGenerator  # Reusing data split logic
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all project files (stgcn.py, spil.py, PoseCon3d.py, preprocessing.py, train.py) are in the same directory.")
    exit(1)

# Constants
HEIGHT = 128
WIDTH = 128
CHANNELS = 17
FRAME_COUNT = 10
BATCH_SIZE = 1  # For benchmarking individual video processing time

def load_stgcn_model(config):
    # As defined in train.py
    output_signature = stgcn.create_skeleton_graph_spec_with_label()
    model = stgcn.ST_GCN(output_signature, num_gcn_layers=config.get("layers", 2))
    path = config["weights_path"]
    if os.path.exists(path):
        print(f"Loading weights from {path}")
        model.load_weights(path)
    return model, config["name"]

def load_poseconv3d_model(config):
    n_frames = config.get("n_frames", 10)
    width = config.get("width", WIDTH)
    height = config.get("height", HEIGHT)
    model = Pose3D(n_frames, height, width, CHANNELS)
    path = config["weights_path"]
    if os.path.exists(path):
        print(f"Loading weights from {path}")
        model.load_weights(path, skip_mismatch=True)
    return model, config["name"]

def load_spil_model(config):
    path = config["weights_path"]
    name = config["name"]
    n_points = config.get("n_points", 1024)
    
    if os.path.exists(path):
        model = spil.ViolenceRecognitionNet(num_classes=2)
        dummy_input = tf.zeros((1, n_points, 4))
        model(dummy_input)
        
        try:
            model.load_weights(path)
            print(f"Weights loaded successfully from {path}")
        except Exception as e_weights:
            print(f"Warning: Exact weight loading failed: {e_weights}")
            print("Attempting to load weights with skip_mismatch=True...")
            model.load_weights(path, skip_mismatch=True)
        return model, name
    
    return spil.ViolenceRecognitionNet(num_classes=2), name

def get_test_generator(config, test_pairs):
    model_type = config["type"]
    n_frames = config.get("n_frames", 10)
    
    if "ST_GCN" in model_type:
        output_signature = stgcn.create_skeleton_graph_spec_with_label()
        gen = stgcn.GraphGenerator(test_pairs, training=False, n_frames=n_frames)
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        # ST_GCN expects (features, label) from separate_features_and_label
        ds = ds.map(stgcn.separate_features_and_label)
        return ds.batch(1)
    
    elif "PoseConv3D" in model_type:
        # Using limb_heatmap as in train.py
        from train import limb_heatmap
        width = config.get("width", WIDTH)
        height = config.get("height", HEIGHT)
        output_signature = (tf.TensorSpec(shape=(None, None, None, 17), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int16))
        gen = FrameGenerator(test_pairs, training=False, n_frames=n_frames, 
                            feature_extractor=limb_heatmap, 
                            output_size=(height, width, CHANNELS))
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        return ds.batch(1)
    
    elif "SPIL" in model_type:
        n_points = config.get("n_points", 1024)
        gen = spil.SPILGenerator(test_pairs, n_frames=n_frames, training=False, n_points=n_points)
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(tf.TensorSpec(shape=(n_points, 4), dtype=tf.float32), 
                             tf.TensorSpec(shape=(), dtype=tf.int32))
        )
        return ds.batch(1)
    
    return None

def run_benchmark(config, test_set):
    # Clear session to reset layer names and avoid "Layer expected 2 variables, received 0" errors
    tf.keras.backend.clear_session()
    
    model_type = config["type"]
    n_frames = config.get("n_frames", 10)
    print(f"\n--- Starting Benchmark for {config['name']} ({model_type}) ---")

    # 1. Load Model
    if "ST_GCN" in model_type:
        model, name = load_stgcn_model(config)
    elif "PoseConv3D" in model_type:
        model, name = load_poseconv3d_model(config)
    elif "SPIL" in model_type:
        model, name = load_spil_model(config)
    else:
        print(f"Invalid model type: {model_type}")
        return None

    # 2. Setup Generator
    test_ds = get_test_generator(config, test_set)
    
    # 4. Benchmarking
    total_prep_time = 0
    total_inf_time = 0
    video_count = 0
    
    print(f"Processing {len(test_set)} videos...")
    
    # Create an iterator to measure prep time manually for each item
    it = iter(test_ds)
    
    for i in range(len(test_set)):
        try:
            # Measure Data Prep Time (Generator execution)
            start_prep = time.time()
            data, label = next(it)
            end_prep = time.time()
            
            prep_time = end_prep - start_prep
            
            # Measure Inference Time
            start_inf = time.time()
            _ = model.predict(data, verbose=0)
            end_inf = time.time()
            
            inf_time = end_inf - start_inf
            
            total_prep_time += prep_time
            total_inf_time += inf_time
            video_count += 1
            
            if (i + 1) % 10 == 0:
                avg_p = total_prep_time / video_count
                avg_i = total_inf_time / video_count
                print(f"[{i+1}/{len(test_set)}] Avg Prep: {avg_p:.3f}s | Avg Inf: {avg_i:.3f}s | Total: {avg_p+avg_i:.3f}s")
                
        except StopIteration:
            break
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Summarizing current results...")
            break
        except Exception as e:
            print(f"Error processing video {i}: {e}")
            continue

    # 5. Summary
    if video_count > 0:
        avg_prep = total_prep_time / video_count
        avg_inf = total_inf_time / video_count
        total_avg = avg_prep + avg_inf
        fps = (video_count * n_frames) / (total_prep_time + total_inf_time)
        
        print("\n" + "="*40)
        print(f"BENCHMARK SUMMARY: {name}")
        print("="*40)
        print(f"Total Videos Processed: {video_count}")
        print(f"Avg Preparation Time:  {avg_prep:.4f} s / video")
        print(f"Avg Inference Time:    {avg_inf:.4f} s / video")
        print(f"Total Avg Time:        {total_avg:.4f} s / video")
        print(f"Throughput (Video):    {1.0/total_avg:.2f} videos/s")
        print(f"Throughput (Frames):   {fps:.2f} FPS")
        print("="*40)

        return {
            "Model": name,
            "Type": model_type,
            "Frames": n_frames,
            "Avg Prep (s)": f"{avg_prep:.4f}",
            "Avg Inf (s)": f"{avg_inf:.4f}",
            "Total Avg (s)": f"{total_avg:.4f}",
            "Throughput (v/s)": f"{1.0/total_avg:.2f}",
            "FPS": f"{fps:.2f}"
        }
    else:
        print("No videos were processed.")
        return None

def save_to_csv(results, filename="benchmark_results.csv"):
    if not results:
        return
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Pose-based Violence Recognition Models")
    parser.add_argument("--all", action="store_true", help="Run benchmark for all models in config")
    parser.add_argument("--model", type=str, help="Specific model name from config to benchmark")
    
    args = parser.parse_args()
    
    # Ensure yolo doesn't print too much
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
    # Load Data once
    print("Loading dataset...")
    dataset = get_dataset('./Real Life Violence Dataset')
    test_set = dataset['test']
    print(f"Test set size: {len(test_set)} videos")

    results = []
    if args.all:
        for config in MODELS_CONFIG:
            res = run_benchmark(config, test_set)
            if res:
                results.append(res)
    elif args.model:
        config = next((m for m in MODELS_CONFIG if m["name"] == args.model), None)
        if config:
            res = run_benchmark(config, test_set)
            if res:
                results.append(res)
        else:
            print(f"Model '{args.model}' not found in configuration.")
    else:
        # Default to first model if nothing specified
        res = run_benchmark(MODELS_CONFIG[0], test_set)
        if res:
            results.append(res)

    if results:
        save_to_csv(results)
