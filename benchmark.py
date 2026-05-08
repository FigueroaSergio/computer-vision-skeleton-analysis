import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import time
import tensorflow as tf
import csv
from model_config import MODELS_CONFIG


# Import existing components
try:
    from STCGN.model import create_skeleton_graph_spec_with_label
    from STCGN.generator import GraphGenerator, separate_features_and_label

    from SPIL.generator import SPILGenerator
    from SPIL.model import ViolenceRecognitionNet

    from PoseConv3D.model import Pose3D
    from PoseConv3D.generator import GeneratorPoseConv3D
    
    from dataset import get_dataset
    from loader import load_model

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all project files (stgcn.py, py, PoseCon3d.py, preprocessing.py, train.py) are in the same directory.")
    exit(1)

# Constants
HEIGHT = 128
WIDTH = 128
CHANNELS = 17
FRAME_COUNT = 10
BATCH_SIZE = 1  # For benchmarking individual video processing time


def get_test_generator(config, test_pairs):
    """
    This function allowe to retrive the  dataset generator for each model type.
    The use of generator allow to reduce the memory usage and the processing time
    because it only processes one video at a time.
    
    Args:
        config: Configuration dictionary for the model.
        test_pairs: List of (video_path, label) pairs for testing.
        
    Returns:
        tf.data.Dataset: Dataset for testing the models.
    """
    model_type = config["type"]
    n_frames = config.get("n_frames", 10)
    
    if "ST_GCN" in model_type:
        output_signature = create_skeleton_graph_spec_with_label()
        gen = GraphGenerator(test_pairs, training=False, n_frames=n_frames)
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        # ST_GCN expects (features, label) from separate_features_and_label
        ds = ds.map(separate_features_and_label)
        return ds.batch(1)
    
    elif "PoseConv3D" in model_type:
        # Using limb_heatmap as in train.py
        from train import limb_heatmap
        width = config.get("width", WIDTH)
        height = config.get("height", HEIGHT)
        output_signature = (tf.TensorSpec(shape=(None, None, None, 17), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int16))
        gen = GeneratorPoseConv3D(test_pairs, training=False, n_frames=n_frames, 
                            feature_extractor=limb_heatmap, 
                            output_size=(height, width, CHANNELS))
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        return ds.batch(1)
    
    elif "SPIL" in model_type:
        n_points = config.get("n_points", 1024)
        gen = SPILGenerator(test_pairs, n_frames=n_frames, training=False, n_points=n_points)
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(tf.TensorSpec(shape=(n_points, 4), dtype=tf.float32), 
                             tf.TensorSpec(shape=(), dtype=tf.int32))
        )
        return ds.batch(1)
    
    return None

def run_benchmark(config, test_set):
    """
    This function benchmark the models by processing a set of videos 
    and measuring the time it takes for each model to process them.

    Note: Since all the models use YOLOv11-pose for the initial feature extraction 
    the benchmark does not take into account this time insted of it, 
    the benchmark measures the total time in two stages: 
    1. Data preparation time (from raw video to model input).
    2. Inference time (from model input to final prediction).
    """
    # Clear session to reset layer names and avoid "Layer expected 2 variables, received 0" errors
    tf.keras.backend.clear_session()
    
    model_type = config["type"]
    n_frames = config.get("n_frames", 10)
    print(f"\n--- Starting Benchmark for {config['name']} ({model_type}) ---")

    # 1. Load Model
    model_data = load_model(config)
    if model_data == None:
        print(f"Invalid model type: {model_type}")
        return None
    model, name = model_data
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
