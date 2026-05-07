import numpy as np
import random
from preprocessing import get_frames, get_class_ids

from .feature_extractor import get_features_spil_from_yolo_results

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")

N_POINTS= 1024

class SPILGenerator:
    def __init__(self, pairs, training=False, n_frames=15, n_points=N_POINTS):
        self.pairs = pairs
        self.training = training
        self.n_frames = n_frames
        self.n_points = n_points

    def __call__(self):
        if self.training:
            random.shuffle(self.pairs)

        for path, name in self.pairs:
            points = self.get_features_spil(path)
            
            # --- FIX: Handle empty detection case ---
            if points.shape[0] == 0:
                # Option A: Skip the video (Common in training)
                if self.training:
                    continue 
                # Option B: Provide a zero-padded "empty" point cloud (Ensures batch consistency)
                else:
                    sampled_points = np.zeros((self.n_points, 4), dtype=np.float32)
            else:
                # Normal Sampling Logic (Section 4.1: N=2048)
                indices = np.random.choice(
                    len(points), 
                    self.n_points, 
                    replace=len(points) < self.n_points
                )
                sampled_points = points[indices]

            label = get_class_ids(name)
            yield sampled_points, label

    def get_features_spil(self, path_video):
        # Your provided frame extraction logic
        frames = get_frames(path_video, self.n_frames)
        results_list = [modelYolo(frame, verbose=False) for frame in frames]
        return get_features_spil_from_yolo_results(results_list)

