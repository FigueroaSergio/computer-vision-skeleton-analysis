import numpy as np
import random
import gc

from .feature_extractor  import get_feautures, format_frames, joint_heatmap
from preprocessing import get_frames, get_class_ids

FRAME_COUNT=5
HEIGHT = 128
WIDTH = 128
CHANNELS = 17

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

class GeneratorPoseConv3D:
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