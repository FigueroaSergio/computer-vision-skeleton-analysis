import numpy as np

def get_features_spil_from_yolo_results(results_list):
    """
    Transforms YOLOv11-pose detections into a 3D point cloud.
    
    The function flattens the predictions from multiple frames into a single list of points.
    Each point is represented as a 4D vector: [x, y, frame_index, confidence].
    This captures the spatial and temporal distribution of joints in a single point set.
    
    Args:
        results_list: List of YOLO results for each frame.
        
    Returns:
        A numpy array of shape (N, 4) containing all detected points.
    """
    all_points = [] 
    for time_idx, results in enumerate(results_list):
        for result in results:
            # Extracting keypoints and confidence as initial features (Section 3.1)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kpts = result.keypoints.data.cpu().numpy() 
                for person in kpts:
                    for joint in person:
                        # x, y, confidence
                        x, y, conf = joint
                        # Formulating 3D point cloud: z is frame index (Source: 137)
                        all_points.append([x, y, float(time_idx), conf])

    # If no points found at all, return empty array with correct feature width
    if len(all_points) == 0:
        return np.empty((0, 4), dtype=np.float32)
        
    return np.array(all_points, dtype=np.float32)