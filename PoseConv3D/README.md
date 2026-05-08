# PoseConv3D

PoseConv3D represents skeleton data as 3D heatmaps, enabling the application of standard 3D Convolutional Neural Networks (CNNs) for action recognition.

## Heatmap Generation

The feature extractor contains three different functions to generate heatmaps from YOLO predictions:

1. **Joint Heatmaps**: Generates Gaussian heatmaps centered at each detected joint position.
2. **Limb Heatmaps**: Generates heatmaps along the limbs (bones) between joints.
3. **Joint and Limb Heatmaps**: A combined representation that merges both joint and limb information.

## Auxiliary Function

Includes a utility function to merge all 17 individual joint heatmaps into a single 3D volume. This was primarily used for:

- **Testing**: Verifying that the resulting heatmaps are well-formatted.
- **Validation**: Ensuring the input volume is suitable for training the 3D CNN model.

## Component Task

1. **Generator**: Manages temporal sampling and batching of 3D heatmaps.
2. **Feature Extractor**: Transforms YOLO keypoints into 3D volumes (heatmaps).
3. **Model**: A 3D CNN architecture (e.g., based on [I3D](https://www.tensorflow.org/tutorials/video/video_classification) ) for processing the heatmap volumes.
