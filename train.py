import io
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
from ultralytics import YOLO
from PoseCon3d import Pose3D 
import stgcn
import spil
from preprocessing import get_frames, get_class_ids
os.environ["WANDB_API_KEY"] = "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"

path_video='./Real Life Violence Dataset/NonViolence/NV_1.mp4'

FRAME_COUNT =5
IMG_SIZE = 640
STEP= 5

frames = get_frames(path_video)
print(len(frames))
modelYolo = YOLO("yolo11n-pose.pt")

HEIGHT= 128
WIDTH = 128
CHANNELS = 17
def get_class(fname):
  if'NonViolence' in fname  :
    return 'NonViolence'
  else:
    return 'Violence'


def get_files_per_class(files):
  """ Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Returns:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class


def split_class_lists(files_for_class, count):
  """ Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Returns:
      Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

from pathlib import Path

def list_all_files_pathlib(directory_path):
    """
    Recursively lists all files in a given directory using pathlib.

    Args:
        directory_path (str): The path to the starting directory.

    Returns:
        list: A list of full file paths as Path objects.
    """
    # Create a Path object from the string path
    directory = Path(directory_path)

    # rglob('*') recursively finds all files and directories
    return [str(path) for path in directory.rglob('*') if path.is_file()]

print(len(list_all_files_pathlib('./Real Life Violence Dataset')))
import json
def get_dataset(path, train=0.7, test=0.2, val=0.1, cache_file='dataset_cache.json'):
  """
  Load datasets from cache if it exists, otherwise create and save it.
  
  Args:
    files: List of file paths
    train: Training split proportion
    test: Test split proportion
    val: Validation split proportion
    cache_file: Path to cache file
    
  Returns:
    Dictionary with train, val, test datasets
  """
  files = list_all_files_pathlib(path)
  # Check if cache exists
  if os.path.exists(cache_file):
    print('file from cache')
    with open(cache_file, 'r') as f:
      return json.load(f)
  
  # Group files by class
  files_for_class = {}
  for path in files:
    class_name = get_class(path)
    if class_name not in files_for_class:
      files_for_class[class_name] = []
    files_for_class[class_name].append(path)

  dataset = {
      'train': [],
      'test': [],
      'val': []
  }

  # For each class, split according to proportions and add to dataset
  for class_name, class_files in files_for_class.items():
    class_size = len(class_files)
    shuffled_indices = np.random.permutation(class_size)
    train_end = int(train * class_size)
    val_end = int((train + val) * class_size)

    class_files = np.array(class_files)[shuffled_indices]
    train_files = class_files[:train_end]
    val_files = class_files[train_end:val_end]
    test_files = class_files[val_end:]

    # Add (file, class) pairs
    dataset['train'].extend([(f, class_name) for f in train_files])
    dataset['val'].extend([(f, class_name) for f in val_files])
    dataset['test'].extend([(f, class_name) for f in test_files])
  
  # Save to cache
  with open(cache_file, 'w') as f:
    json.dump(dataset, f)
  
  return dataset

dataset = get_dataset('./Real Life Violence Dataset')
print('Train: ', len(dataset['train']))
print('Val: ', len(dataset['val']))
print('Test: ', len(dataset['test']))
import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.login()
import tensorflow as tf
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger
from sklearn.metrics import f1_score, confusion_matrix, classification_report

class WandbDatasetEvalCallback(keras.callbacks.Callback):
    def __init__(self, dataset, class_names=None):
        super().__init__()
        self.dataset = dataset
        self.class_names = class_names
        
        print("Extracting labels from validation dataset (this may take a moment)...")
        y_true_list = []
        
        # Manually iterate to extract labels since len() is unknown
        # We use .as_numpy_iterator() for speed and compatibility
        for _, y in dataset:
            # Handle if labels are provided as (Batch, Classes) or (Batch,)
            label_np = y.numpy()
            if len(label_np.shape) > 1: # One-hot encoded
                y_true_list.extend(np.argmax(label_np, axis=1))
            else: # Integer encoded
                y_true_list.extend(label_np)
        
        self.y_true = np.array(y_true_list)
        print(f"Successfully extracted {len(self.y_true)} validation labels.")

    def on_epoch_end(self, epoch, logs=None):
        # 1. Predict on the dataset
        # verbose=0 keeps the console clean
        y_pred_probs = self.model.predict(self.dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 2. Calculate Weighted F1
        f1_weighted = f1_score(self.y_true, y_pred, average='weighted')
        
        # 3. Create Classification Report (Per-Class)
        report = classification_report(
            self.y_true, 
            y_pred, 
            labels=[0, 1],
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )

        # 4. Log to WandB
        metrics = {
            "val/f1_weighted": f1_weighted,
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.y_true, 
                preds=y_pred,
                class_names=self.class_names
            )
        }

        # Dynamically log every class's F1 to track Violence vs Non-Violence
        if self.class_names:
            for cls in self.class_names:
                if cls in report:
                    metrics[f"val/class_{cls}_f1"] = report[cls]['f1-score']

        wandb.log(metrics)
        print(f" — val_f1_weighted: {f1_weighted:.4f}")
def Train(name,model, epochs, train, val, run_id=None,  steps_per_epoch=None):
    run_name = name.replace(" ", "_").lower()
    run = wandb.init(project='computer-vision',name=run_name,  id=run_id,resume="allow")
    wandb.config.update({"frame_count": FRAME_COUNT})
    backup_path = f"training/{name}"
    os.makedirs(backup_path, exist_ok=True)
    model.fit(
        train,
        epochs = epochs,
        validation_data = val,
         steps_per_epoch=steps_per_epoch,
    callbacks=[
        keras.callbacks.ModelCheckpoint(f"models/{name}.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,     
            min_lr=1e-6,  # Minimum learning rate
            verbose=1
        ),
        keras.callbacks.BackupAndRestore(backup_path, save_freq='epoch', delete_checkpoint=True),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=9,   # Wait longer than reduce_lr (7 > 3)
            restore_best_weights=True,
            verbose=1
        ),
        WandbMetricsLogger( log_freq='batch'),
        WandbDatasetEvalCallback(val, class_names=['Violence','NonViolence'])

    ]
    )
    run.finish()
    return  model
# FEATURE extraction

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

# GENERATOR
import random
import tensorflow as tf

def get_feautures(frame, feature_extractor=joint_heatmap, output_size = (HEIGHT, WIDTH,CHANNELS)):
  results = modelYolo(frame, verbose=False)

  if len(results[0])==0 :
    return np.zeros(output_size)
  processed_person = []

  for person in results[0].keypoints:
    # print('Processing person')
    joints = person.data.cpu().numpy()[0]
    h, w, _ = frame.shape
    sigma = 5
    heatmap = feature_extractor(w, h, joints,sigma)
    processed_person.append(heatmap)
  stacked_matrices = np.stack(processed_person)
  averaged_matrix = np.mean(stacked_matrices, axis=0)
  del stacked_matrices
  return averaged_matrix


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
import gc
def frames_from_video_file(video_path, n_frames,  feature_extractor=joint_heatmap, output_size = (HEIGHT, WIDTH,CHANNELS)):
  formatted_frames = []

  frames = get_frames(video_path,n_frames)
  for frame in frames:
    features = get_feautures(frame, feature_extractor, output_size)
    # print(features.shape)
    formatted_frame = format_frames(features, (output_size[0], output_size[1]))
    # print('Formated shape: ',formatted_frame.shape)
    formatted_frames.append(formatted_frame)

  return np.array(formatted_frames)
  
class FrameGenerator:
  def __init__(self, pairs, training = False, n_frames=FRAME_COUNT, feature_extractor=joint_heatmap, output_size=(HEIGHT, WIDTH,CHANNELS)):
    """ Returns a set of frames with their associated label.

      Args:
        paparis:[[path,label]].
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.pairs = pairs
    self.n_frames = n_frames
    self.training = training
    self.output_size = output_size
    self.feature_extractor = feature_extractor


  def __call__(self):


    if self.training:
      random.shuffle(self.pairs)

    for path, name in self.pairs:
      video_frames = frames_from_video_file(path, self.n_frames, self.feature_extractor,self.output_size)
      label = get_class_ids(name) # Encode labels
      yield video_frames, label
      del video_frames
      gc.collect()
    


POSE_CONV3D='POSE_CONV3D'
ST_CGN='ST_CGN'
SPIL='SPIL'
if __name__ == "__main__":
    MODEL = SPIL
    # AUTOTUNE = 0
    BATCH_SIZE=8

    #ONLY JOINTS

    HEIGHT= 128
    WIDTH = 128
    CHANNELS = 17
    FRAME_COUNT =10
    if(MODEL==POSE_CONV3D):
      model = Pose3D(FRAME_COUNT, HEIGHT, WIDTH, CHANNELS)
      keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)
      output_signature = (tf.TensorSpec(
                              shape = (None ,None, None, 17), 
                              dtype = tf.float32
                              ),
                          tf.TensorSpec(shape = (), dtype = tf.int16))
      train_ds = tf.data.Dataset.from_generator(FrameGenerator(dataset['train'],
                                                  training=True, 
                                                  n_frames=FRAME_COUNT, 
                                                  feature_extractor=limb_heatmap, 
                                                  output_size=(HEIGHT, WIDTH,CHANNELS)
                                              ),
                                                output_signature = output_signature)
      val_ds = tf.data.Dataset.from_generator(FrameGenerator(dataset['val'],
                                                  training=True, 
                                                  n_frames=FRAME_COUNT, 
                                                  feature_extractor=limb_heatmap, 
                                                  output_size=(HEIGHT, WIDTH,CHANNELS)
                                                  ),
                                              output_signature = output_signature)
      # train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
      # val_ds = val_ds.prefetch(buffer_size = AUTOTUNE)
      train_ds = train_ds.batch(BATCH_SIZE)
      val_ds = val_ds.batch(BATCH_SIZE)
      # Train('Joints_PoseConv3D',model, 100, train_ds,val_ds,'Joints_PoseConv3D_1' ,steps_per_epoch=steps_per_epoch,)
      Train(f'Limbs_PoseConv3D-f{FRAME_COUNT}',model, 100, train_ds,val_ds,'ba87qs2n')


    ## STGCN
    if(MODEL==ST_CGN):
      LAYERS= 2
      output_signature= stgcn.create_skeleton_graph_spec_with_label()
      train_ds = tf.data.Dataset.from_generator(stgcn.GraphGenerator(dataset['train'],
                                                  training=True, 
                                                  n_frames=FRAME_COUNT, 
                                              ),
                                                output_signature = output_signature)
      val_ds = tf.data.Dataset.from_generator(stgcn.GraphGenerator(dataset['val'],
                                                  training=False, 
                                                  n_frames=FRAME_COUNT, 
                                              ),
                                                output_signature = output_signature)
      # 
      skeleton_gnn_model = stgcn.ST_GCN(output_signature, num_gcn_layers=LAYERS)
      train_ds = train_ds.batch(BATCH_SIZE)
      val_ds = val_ds.batch(BATCH_SIZE)
      # 
      train_ds = train_ds.map(stgcn.separate_features_and_label)
      val_ds = val_ds.map(stgcn.separate_features_and_label)
      num_train_samples=len(dataset['train'])
      steps_per_epoch= num_train_samples // BATCH_SIZE
      Train(f'ST_CGN-f{FRAME_COUNT}-L{LAYERS}',skeleton_gnn_model, 100, train_ds,val_ds)
      # 

    if(MODEL == SPIL):
      N_POINTS= 1024



      train_gen = spil.SPILGenerator(dataset['train'], n_frames=FRAME_COUNT,training=True)
      val_gen = spil.SPILGenerator(dataset['val'],  n_frames=FRAME_COUNT,training=False)

      train_ds = tf.data.Dataset.from_generator(
          train_gen,
          output_signature=(tf.TensorSpec(shape=(N_POINTS, 4), dtype=tf.float32), 
                            tf.TensorSpec(shape=(), dtype=tf.int32))
      ).batch(BATCH_SIZE)

      val_ds = tf.data.Dataset.from_generator(
          val_gen,
          output_signature=(tf.TensorSpec(shape=(N_POINTS, 4), dtype=tf.float32), 
                            tf.TensorSpec(shape=(), dtype=tf.int32))
      ).batch(BATCH_SIZE)
      model = spil.ViolenceRecognitionNet(num_classes=2)
      optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, clipnorm=1.0)
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
      model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
      Train(f'SPIL-f{FRAME_COUNT}',model, 100, train_ds,val_ds )
    # model =  Pose3D(FRAME_COUNT, HEIGHT, WIDTH, CHANNELS)
  

    # model.load_weights('models/Limbs_PoseConv3D-f10.keras',skip_mismatch=True)
    # model.summary()
