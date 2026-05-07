import numpy as np
import cv2

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")

FRAME_COUNT=5
HEIGHT = 128
WIDTH = 128
CHANNELS = 17

def joint_value(i,j,x,y,confidence, sigma):
  """Calculates the value of a joint in a heatmap."""
  return np.exp(-((i - x)**2 + (j - y)**2) / (2 * sigma**2)) * confidence

def draw_joint_heatmap(arr, x, y, confidence, sigma):
    h, w = arr.shape
    bbox_size = int(3 * sigma) # Adjust this multiplier as needed

    # Define the bounding box coordinates
    min_x =int( max(0, x - bbox_size))
    max_x =int( min(w, x + bbox_size))
    min_y =int( max(0, y - bbox_size))
    max_y =int( min(h, y + bbox_size))
    # 0,0 -----> x
    # |
    # |
    # y
    i = np.arange(min_x, max_x, 1, np.float32)
    j = np.arange(min_y, max_y, 1, np.float32)
    if not (len(i) and len(j)):
        return
    j=j[:,None]
    patch = joint_value(i, j, x ,y, confidence, sigma)
    arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y,min_x:max_x], patch)


def joint_heatmap(w,h,joints, sigma):
  """Generates a heatmap for given joints."""
  heatmap = np.zeros((h, w, len(joints)))

  for k, joint in enumerate(joints):
    x, y, confidence = joint[0], joint[1], joint[2]
    draw_joint_heatmap(heatmap[:,:,k],x, y, confidence, sigma)
  return heatmap

def joint_heatmap_numpy(w, h, joints, sigma):
    """Generates a heatmap using NumPy's vectorized operations."""
    heatmap = np.zeros((h, w, len(joints)), dtype=np.float32)

    for k, joint in enumerate(joints):
        x, y, confidence = joint[0], joint[1], joint[2]

        # Create a grid of coordinates
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        # Calculate the distance from the joint
        d = ((yy - y)**2 + (xx - x)**2) / (2 * sigma**2)

        # Apply the Gaussian function
        heatmap[:, :, k] = confidence * np.exp(-d)

    return heatmap

EPS = 1e-3

def draw_limb_heatmap( arr, starts, ends,sigma):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        img_h, img_w = arr.shape
        start_value = starts[2]
        end_value = ends[2]
        end = ends[:2]
        start = starts[:2]

        value_coeff = min(start_value, end_value)

        if value_coeff < EPS:
            return
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = np.arange(min_x, max_x, 1, np.float32)
        y = np.arange(min_y, max_y, 1, np.float32)

        if not (len(x) and len(y)):
            return
        y = y[:, None]

        x_0 = np.zeros_like(x)
        y_0 = np.zeros_like(y)

        # distance to start keypoints
        d2_start = ((x - start[0])**2 + (y - start[1])**2)
        # distance to end keypoints
        d2_end = ((x - end[0])**2 + (y - end[1])**2)
        # the distance between start and end keypoints.
        d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)
        if d2_ab < 1:
            draw_joint_heatmap(arr, start[0],start[1], start_value,sigma)
            return

        coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab
        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate
        position = np.stack([x + y_0, y + x_0], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line
        patch = np.exp(-d2_seg / 2. / sigma**2)
        patch = patch * value_coeff
        arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)

def limb_heatmap(w,h,joints, sigma):
  """Generates a heatmap for given joints."""
  heatmap = np.zeros((h, w, len(joints)), dtype=np.float32)
  skeleton = [
          (0, 1), (0, 2), (1, 3),
          (2, 4), (5, 6), (5, 7),
          (7, 9), (6, 8), (8, 10),
          (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14),
          (14, 16)
    ]
  for i,(start_idx, end_idx) in enumerate(skeleton):
      start = joints[start_idx]
      end = joints[end_idx]
      draw_limb_heatmap(heatmap[:,:,i], start, end, sigma)
  return heatmap

def aggregate_heatmap(heatmap):
  """Aggregates the k layers of the heatmap into a single RGB image."""
  # Sum across the k dimension
  aggregated_heatmap = np.sum(heatmap, axis=2)

  # Normalize to the range [0, 1]
  aggregated_heatmap = (aggregated_heatmap - np.min(aggregated_heatmap)) / (np.max(aggregated_heatmap) - np.min(aggregated_heatmap))

  # Convert to RGB by repeating the channel
  rgb_heatmap = np.stack([aggregated_heatmap] * 3, axis=-1)

  return aggregated_heatmap

def get_frame_features(frame, feature_extractor ):
  results = modelYolo(frame)
  h, w, _ = frame.shape
  if(len(results[0])==0):
    return np.zeros((IMG_SIZE,IMG_SIZE,17)), []
  processed_data = []
  for person in results[0].keypoints:
    joints = person.data.cpu().numpy()[0]
    h, w, _ = frame_uint8.shape
    sigma = 5
    heatmap = feature_extractor(w, h, joints,sigma)
    processed_data.append(heatmap)
  stacked_features = np.stack(processed_data)
  aggregate_features = np.mean(stacked_features, axis=0)
  return aggregate_features, stacked_features


def get_features_conv3d_from_yolo_results(results, frame_shape, feature_extractor=joint_heatmap, output_size = (HEIGHT, WIDTH,CHANNELS)):
  if len(results[0])==0 :
    return np.zeros(output_size)
  processed_person = []

  for person in results[0].keypoints:
    # print('Processing person')
    joints = person.data.cpu().numpy()[0]
    h, w, _ = frame_shape
    sigma = 5
    heatmap = feature_extractor(w, h, joints,sigma)
    processed_person.append(heatmap)
  stacked_matrices = np.stack(processed_person)
  averaged_matrix = np.mean(stacked_matrices, axis=0)
  del stacked_matrices
  return averaged_matrix

def get_feautures(frame, feature_extractor=joint_heatmap, output_size = (HEIGHT, WIDTH,CHANNELS)):
  results = modelYolo(frame, verbose=False)
  return get_features_conv3d_from_yolo_results(results, frame.shape, feature_extractor, output_size)


def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame  = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
  return frame