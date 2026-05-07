import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import tensorflow as tf

try:
    from STCGN.model import ST_GCN, create_skeleton_graph_spec_with_label

    from SPIL.model import ViolenceRecognitionNet

    from PoseConv3D.model import Pose3D
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all project files (stgcn.py, py, PoseCon3d.py, preprocessing.py, train.py) are in the same directory.")
    exit(1)

# Constants
HEIGHT = 128
WIDTH = 128
CHANNELS = 17
FRAME_COUNT = 10

def load_stgcn_model(config):
    # As defined in train.py
    output_signature = create_skeleton_graph_spec_with_label()
    model = ST_GCN(output_signature, num_gcn_layers=config.get("layers", 2))
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
        model = ViolenceRecognitionNet(num_classes=2)
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
    
    return ViolenceRecognitionNet(num_classes=2), name

def load_model(config):

    model_type = config["type"]
    n_frames = config.get("n_frames", 10)


    # 1. Load Model
    if "ST_GCN" in model_type:
        return load_stgcn_model(config)
    elif "PoseConv3D" in model_type:
        return load_poseconv3d_model(config)
    elif "SPIL" in model_type:
        return load_spil_model(config)
    else:
        print(f"Invalid model type: {model_type}")
        return None