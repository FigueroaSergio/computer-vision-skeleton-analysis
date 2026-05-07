import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["WANDB_API_KEY"] = "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"
import numpy as np
from ultralytics import YOLO

from PoseConv3D.model import Pose3D 
from PoseConv3D.feature_extractor import limb_heatmap
from PoseConv3D.generator import GeneratorPoseConv3D


from STCGN.model import ST_GCN,create_skeleton_graph_spec_with_label
from STCGN.generator import GraphGenerator, separate_features_and_label

from SPIL.model import ViolenceRecognitionNet
from SPIL.generator import SPILGenerator


FRAME_COUNT =5
IMG_SIZE = 640
STEP= 5

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

# dataset = get_dataset('./Real Life Violence Dataset')
# print('Train: ', len(dataset['train']))
# print('Val: ', len(dataset['val']))
# print('Test: ', len(dataset['test']))
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
    save_extension='keras'
    if('ST_GCN' in name or 'ST_CGN' in name):
       save_extension='h5'
    model.fit(
        train,
        epochs = epochs,
        validation_data = val,
         steps_per_epoch=steps_per_epoch,
    callbacks=[
        keras.callbacks.ModelCheckpoint(f"models/{name}.{save_extension}", save_best_only=True),
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
 


POSE_CONV3D='PoseConv3D'
ST_CGN='ST_GCN'
SPIL='SPIL'
if __name__ == "__main__":
    import argparse
    from model_config import MODELS_CONFIG

    parser = argparse.ArgumentParser(description="Train Violence Recognition Models")
    parser.add_argument("--config", type=str, help="Model name from config (e.g. SPIL-f10)")
    parser.add_argument("--model", type=str, choices=[POSE_CONV3D, ST_CGN, SPIL], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames per video")
    parser.add_argument("--layers", type=int, default=2, help="Number of GCN layers (for ST_GCN)")
    parser.add_argument("--n_points", type=int, default=1024, help="Number of points (for SPIL)")
    parser.add_argument("--dataset_path", type=str, default='./Real Life Violence Dataset', help="Path to the dataset")
    parser.add_argument("--resume_id", type=str, default=None, help="WandB run ID to resume")

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        config = next((m for m in MODELS_CONFIG if m["name"] == args.config), None)
        if not config:
            print(f"Error: Model configuration '{args.config}' not found in model_config.py")
            exit(1)
        
        # Override defaults with config values
        MODEL_TYPE = config["type"]
        FRAME_COUNT = config.get("n_frames", 10)
        LAYERS = config.get("layers", 2)
        N_POINTS = config.get("n_points", 1024)
        NAME = config["name"]
    else:
        if not args.model:
            print("Error: Either --config or --model must be specified.")
            parser.print_help()
            exit(1)
        MODEL_TYPE = args.model
        FRAME_COUNT = args.frames
        LAYERS = args.layers
        N_POINTS = args.n_points
        if MODEL_TYPE == ST_CGN:
            NAME = f'ST_CGN-f{FRAME_COUNT}-L{LAYERS}'
        elif MODEL_TYPE == POSE_CONV3D:
            NAME = f'Limbs_PoseConv3D-f{FRAME_COUNT}'
        else:
            NAME = f'SPIL-f{FRAME_COUNT}'

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATASET_PATH = args.dataset_path
    RESUME_ID = args.resume_id

    print(f"Training configuration: {NAME}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Frames: {FRAME_COUNT}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

    # Load Dataset
    dataset = get_dataset(DATASET_PATH)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['val'])} val, {len(dataset['test'])} test")

    if MODEL_TYPE == POSE_CONV3D:
        # CONV3D settings
        HEIGHT = 128
        WIDTH = 128
        CHANNELS = 17
        
        model = Pose3D(FRAME_COUNT, HEIGHT, WIDTH, CHANNELS)
        output_signature = (
            tf.TensorSpec(shape=(None, None, None, 17), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16)
        )
        
        train_ds = tf.data.Dataset.from_generator(
            GeneratorPoseConv3D(dataset['train'], training=True, n_frames=FRAME_COUNT, 
                           feature_extractor=limb_heatmap, output_size=(HEIGHT, WIDTH, CHANNELS)),
            output_signature=output_signature
        ).batch(BATCH_SIZE)
        
        val_ds = tf.data.Dataset.from_generator(
            GeneratorPoseConv3D(dataset['val'], training=False, n_frames=FRAME_COUNT, 
                           feature_extractor=limb_heatmap, output_size=(HEIGHT, WIDTH, CHANNELS)),
            output_signature=output_signature
        ).batch(BATCH_SIZE)

        Train(NAME, model, EPOCHS, train_ds, val_ds, run_id=RESUME_ID)

    elif MODEL_TYPE == ST_CGN:
        output_signature = create_skeleton_graph_spec_with_label()
        
        train_ds = tf.data.Dataset.from_generator(
            GraphGenerator(dataset['train'], training=True, n_frames=FRAME_COUNT),
            output_signature=output_signature
        ).batch(BATCH_SIZE).map(separate_features_and_label)
        
        val_ds = tf.data.Dataset.from_generator(
            GraphGenerator(dataset['val'], training=False, n_frames=FRAME_COUNT),
            output_signature=output_signature
        ).batch(BATCH_SIZE).map(separate_features_and_label)

        skeleton_gnn_model = ST_GCN(output_signature, num_gcn_layers=LAYERS)
        
        num_train_samples = len(dataset['train'])
        steps_per_epoch = num_train_samples // BATCH_SIZE
        
        Train(NAME, skeleton_gnn_model, EPOCHS, train_ds, val_ds, run_id=RESUME_ID, steps_per_epoch=steps_per_epoch)

    elif MODEL_TYPE == SPIL:
        train_gen = SPILGenerator(dataset['train'], n_frames=FRAME_COUNT, training=True)
        val_gen = SPILGenerator(dataset['val'], n_frames=FRAME_COUNT, training=False)

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

        model = ViolenceRecognitionNet(num_classes=2)
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, clipnorm=1.0)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        Train(NAME, model, EPOCHS, train_ds, val_ds, run_id=RESUME_ID)
