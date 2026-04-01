import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from preprocessing import get_frames,get_class_ids

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")


def get_neighbors(points, k=20):
    """
    Finds k-nearest neighbors for each point to form local regions.
    [cite_start]Based on the paper's strategy to group local regional points[cite: 7].
    
    Args:
        [cite_start]points: (batch_size, N, 3) - Coordinates (x, y, t) [cite: 82]
    Returns:
        grouped_indices: (batch_size, N, k)
    """
    # Calculate pairwise distances
    # Using k-NN here as a standard proxy for the "local region" grouping described [cite: 108]
    # [cite_start]In the paper, radius search is also mentioned[cite: 109].
    dist = tf.reduce_sum(points**2, axis=2, keepdims=True) - \
           2 * tf.matmul(points, points, transpose_b=True) + \
           tf.transpose(tf.reduce_sum(points**2, axis=2, keepdims=True), [0, 2, 1])
    
    # Retrieve indices of the k nearest neighbors
    _, indices = tf.math.top_k(-dist, k=k)
    return indices
def index_points(points, idx):
    """
    Gathers points based on indices.
    [cite_start]Used to retrieve neighbor coordinates and features[cite: 120].
    """
    # batch_size = tf.shape(points)[0]
    # num_points = tf.shape(points)[1]
    # data_dim = tf.shape(points)[2]
    # k = tf.shape(idx)[2]

    # batch_idx = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, num_points, k))
    # idx = tf.stack([batch_idx, tf.tile(tf.reshape(tf.range(num_points), (1, -1, 1)), (batch_size, 1, k)), idx], axis=-1)
    
    return tf.gather(points, idx,batch_dims=1)

class SPIL_Layer(layers.Layer):
    def __init__(self, out_channels, num_heads=8, d_threshold=0.04, **kwargs):
        """
        Initializes the SPIL Layer.
        
        Args:
            out_channels: Output feature dimension.
            num_heads: Number of heads for multi-head mechanism.
            d_threshold: Distance threshold 'd' for SPIL-Mask.
        """
        super(SPIL_Layer, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.d_threshold = d_threshold
        self.head_dim = out_channels // num_heads

        # Learnable linear projections for Feature Term (phi and theta in Eq 3) 
        self.phi = layers.Dense(self.head_dim, activation='relu')
        self.theta = layers.Dense(self.head_dim, activation='relu')
        
        # Learnable MLPs for Position Term (M1, M2 in Eq 5/6) 
        self.m1 = layers.Dense(self.head_dim // 2, activation='relu')
        self.m2 = layers.Dense(self.head_dim // 2, activation='relu')
        
        # Linear projection psi for Position Term 
        self.psi = layers.Dense(1, activation='relu')

        # Layer-specific weight matrix Z for convolution
        self.Z_weight = layers.Dense(self.head_dim)

    def call(self, center_xyz, center_features, neighbor_xyz, neighbor_features):
        """
        Performs the SPIL convolution.
        
        Args:
            center_xyz: (B, N, 3) - Centroid coordinates
            center_features: (B, N, C) - Centroid features
            neighbor_xyz: (B, N, K, 3) - Neighbor coordinates
            neighbor_features: (B, N, K, C) - Neighbor features
        """
        
        # Multi-head Mechanism Loop [cite: 187] ---
        # Note: In efficient TF, we process heads in parallel via reshaping, 
        # but for clarity regarding the paper's math, we treat the dimensions explicitly.
        
        # 1. Feature Term R^F
        # Project features
        feat_i = self.phi(center_features)  # (B, N, head_dim)
        feat_j = self.theta(neighbor_features) # (B, N, K, head_dim)
        # print(f"center_xyz shape: {center_xyz.shape}")
        # print(f"center_features shape: {feat_i.shape}")
        # print(f"neighbor_xyz shape: {feat_j.shape}")
        # print(f"neighbor_features shape: {feat_j.shape}")
        
        # Expand feat_i for broadcasting: (B, N, 1, head_dim)
        feat_i_exp = tf.expand_dims(feat_i, axis=2)
        
        # Calculate dot product similarity for R^F
        # R^F = phi(p_i)^T * theta(p_j)
        r_f = tf.reduce_sum(feat_i_exp * feat_j, axis=-1, keepdims=True) # (B, N, K, 1)

        # 2. Position Term R^L using SPIL-Mask (Eq 6)
        # Encode positions using MLPs
        pos_i_encoded = self.m1(center_xyz) # (B, N, dim/2)
        pos_j_encoded = self.m2(neighbor_xyz) # (B, N, K, dim/2)
        
        # Concatenate encoded positions (||)
        pos_i_exp = tf.expand_dims(pos_i_encoded, axis=2) # (B, N, 1, dim/2)
        pos_i_tiled = tf.tile(pos_i_exp, [1, 1, tf.shape(neighbor_xyz)[2], 1])
        pos_concat = tf.concat([pos_i_tiled, pos_j_encoded], axis=-1)
        
        # Project to scalar using psi
        r_l_base = self.psi(pos_concat) # (B, N, K, 1)

        # Apply Masking Strategy 
        # Condition: if (l_i^z == l_j^z) and distance > d
        
        t_i = tf.expand_dims(center_xyz[..., 2], axis=2) # (B, N, 1)
        # [..., 2] is exactly the same as writing center_xyz[:, :, 2]
        # just go to the very last dimension and grab the value at index 2
        # tf.expand_dims is used to add a dimension of size 1 to a tensor at a specific location.
        # center_xyz shape: (2, 2, 3)
        # [
        #   Batch 0: [ [x0, y0, z0], [x1, y1, z1] ],
        #   Batch 1: [ [x2, y2, z2], [x3, y3, z3] ]
        # ]
        # Result of slicing: [...,2] (2,2)
        # [
        #   Batch 0: [z0, z1],
        #   Batch 1: [z2, z3]
        # ]
        # t_i shape: (2, 2, 1) = expand_dims (2, 2, 1)
        # [
        #   Batch 0: [ [z0], [z1] ],
        #   Batch 1: [ [z2], [z3] ]
        # ]



        t_j = neighbor_xyz[..., 2] # (B, N, K)
        
        # Check same frame
        same_frame = tf.cast(tf.equal(t_i, t_j), tf.float32)
        
        # Check distance
        center_xyz_exp = tf.expand_dims(center_xyz, axis=2)
        euclidean_dist = tf.norm(center_xyz_exp - neighbor_xyz, axis=-1)
        dist_mask = tf.cast(euclidean_dist > self.d_threshold, tf.float32)


        
        
        # Final Mask: 1 if we should mask (set to 0), 0 otherwise
        # The paper says: R^L = 0 if condition met.
        mask_condition = same_frame * dist_mask 
        
        # Apply mask: if condition is 1, result is 0. Else result is r_l_base.
        r_l = r_l_base * (1.0 - tf.expand_dims(mask_condition, axis=-1))

        # 3. Compute Weights W_ij (Eq 2) [cite: 129]
        # Numerator: R^L * exp(R^F)
        # 1. Scale the dot product (crucial for stability)
        r_f = r_f / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        r_f_max = tf.reduce_max(r_f, axis=2, keepdims=True)
        numerator = r_l * tf.exp(r_f- r_f_max)
        
        # Denominator: Sum over neighbors K
        denominator = tf.reduce_sum(numerator, axis=2, keepdims=True) + 1e-8
        
        # Interaction Weights
        W_ij = numerator / denominator # (B, N, K, 1)

        # 4. Convolution Update (Eq 7 & 8) [cite: 204]
        # X^(l+1) = W * X^(l) * Z
        # Transform neighbor features with layer-specific weight Z first
        transformed_features = self.Z_weight(neighbor_features) # (B, N, K, head_dim)
        
        # Apply interaction weights
        weighted_features = W_ij * transformed_features
        
        # Aggregate over neighbors (Summation implies convolution over K)
        local_aggregated = tf.reduce_sum(weighted_features, axis=2) # (B, N, head_dim)
        
        return local_aggregated

class MultiHeadSPIL(layers.Layer):
    def __init__(self, out_channels, num_heads=8, d_threshold=0.04,**kwargs):
        """
        Wrapper for Multi-Head Mechanism.
        [cite_start]Paper: "Multiple heads attend to fuse features... to jointly handle different types of relationships"[cite: 8].
        """
        super(MultiHeadSPIL, self).__init__(**kwargs)
        self.heads = []
        self.num_heads = num_heads
        # [cite_start]Create independent heads [cite: 191]
        for _ in range(num_heads):
            self.heads.append(SPIL_Layer(out_channels, num_heads, d_threshold))
            
    def call(self, center_xyz, center_features, neighbor_xyz, neighbor_features):
        head_outputs = []
        for head in self.heads:
            # Each head computes features independently
            out = head(center_xyz, center_features, neighbor_xyz, neighbor_features)
            head_outputs.append(out)
            
        # [cite_start]Concatenate results from all heads (Eq 8) [cite: 204]
        # Concatenation happens at the feature dimension
        return tf.concat(head_outputs, axis=-1)

class ViolenceRecognitionNet(Model):
    def __init__(self, num_classes=2):
        """
        Architecture Overview:
        [cite_start]Input -> SPIL Module x3 -> MLP & Pooling -> Global Feature -> Classifier[cite: 95].
        """
        super(ViolenceRecognitionNet, self).__init__()
        
        # Hyperparameters
        self.k_neighbors = 20 # Defined by K neighbors [cite: 102]
        self.num_heads = 8    # Empirically set to 8 [cite: 257]
        self.d_thresh = 0.04  # Empirically set to 0.04 [cite: 257]
        
        # SPIL Modules (3 layers) [cite: 231]
        # Increasing feature dimensions as is typical in PointNet-like architectures
        self.spil1 = MultiHeadSPIL(out_channels=32, num_heads=self.num_heads, d_threshold=self.d_thresh)
        self.spil2 = MultiHeadSPIL(out_channels=64, num_heads=self.num_heads, d_threshold=self.d_thresh)
        self.spil3 = MultiHeadSPIL(out_channels=128, num_heads=self.num_heads, d_threshold=self.d_thresh)
        
        # MLP before pooling
        self.mlp_final = layers.Dense(256, activation='relu') # [cite: 211]
        
        # Global Average Pooling [cite: 211]
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Classification Head
        self.dropout = layers.Dropout(0.4) # Dropout ratio 0.4 [cite: 229]
        self.classifier = layers.Dense(num_classes, activation='softmax') # [cite: 212]

    def call(self, inputs):
        """
        Args:
            [cite_start]inputs: (Batch, N, 3+C) - Matrix of N points with 3-dim coords and C-dim features [cite: 84]
                    We assume the first 3 channels are (x, y, t) coordinates.
        """
        # Separate Coordinates and Features
        # [cite_start]Input is N x (3+C) [cite: 84]
        xyz = inputs[:, :, :3] 
        features = inputs[:, :, :3] 
        
        # print('xyz', xyz.shape)
        # print('features', features.shape)
        
        # [cite_start]If no initial features provided, use coordinates or confidence as features [cite: 120]
        # if tf.shape(features)[-1] == 0:
        #     features = xyz

        # --- Layer 1 ---
        # Find Neighbors
        idx = get_neighbors(xyz, k=self.k_neighbors)
        neighbor_xyz = index_points(xyz, idx)
        neighbor_feat = index_points(features, idx)
        
        # Apply SPIL
        features = self.spil1(xyz, features, neighbor_xyz, neighbor_feat)
        
        # --- Layer 2 ---
        idx = get_neighbors(xyz, k=self.k_neighbors)
        neighbor_xyz = index_points(xyz, idx)
        neighbor_feat = index_points(features, idx)
        features = self.spil2(xyz, features, neighbor_xyz, neighbor_feat)
        
        # --- Layer 3 ---
        idx = get_neighbors(xyz, k=self.k_neighbors)
        neighbor_xyz = index_points(xyz, idx)
        neighbor_feat = index_points(features, idx)
        features = self.spil3(xyz, features, neighbor_xyz, neighbor_feat)
        
        # --- Classification ---
        # MLP & Pooling
        x = self.mlp_final(features)
        
        # [cite_start]Global Pooling (Average) [cite: 211]
        x = self.global_pool(x)
        
        # Dropout and Predict
        x = self.dropout(x)
        output = self.classifier(x)
        
        return output
import tensorflow as tf
import numpy as np
import random
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
        all_points = [] 

        for time_idx, frame in enumerate(frames):
            results = modelYolo(frame, verbose=False)

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