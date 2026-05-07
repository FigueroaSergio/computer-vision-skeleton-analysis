
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import random
from preprocessing import get_class_ids
from STCGN.feature_extractor import graph_from_video

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")


# Utilizaremos la especificación del grafo que definimos previamente
FRAME_COUNT =10
HEIGHT= 128
WIDTH = 128
CHANNELS = 17
FRAME_COUNT =10


class GraphGenerator:
  def __init__(self, pairs, training = False, n_frames=FRAME_COUNT ):
    """ Returns a set of frames with their associated label.

      Args:
        paparis:[[path,label]].
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.pairs = pairs
    self.n_frames = n_frames
    self.training = training


  def __call__(self):


    if self.training:
      random.shuffle(self.pairs)

    for path, name in self.pairs:
      graph = graph_from_video(path,name, self.n_frames)
      label = get_class_ids(name)
      yield graph


def separate_features_and_label(graph_tensor):
    """
    Extracts the 'label' from the graph context to be the Y value.
    Returns: (features, label)
    """
    # X data is the GraphTensor itself
    features = graph_tensor
    
    # Y data is the label from the context
    label = graph_tensor.context['label']
    
    # If using BinaryCrossentropy, ensure label is float32 (optional, but good practice)
    # label = tf.cast(label, tf.float32) 
    label = tf.squeeze(graph_tensor.context['label'], axis=-1)
    # Keras expects (features, target)
    return features, label