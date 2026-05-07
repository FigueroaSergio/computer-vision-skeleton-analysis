
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_gnn as tfgnn
from preprocessing import get_frames,get_class_ids
import numpy as np

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")




def get_limbs_person(person_number):
    skeleton = [
          (0, 1), (0, 2), (1, 3),
          (2, 4), (5, 6), (5, 7),
          (7, 9), (6, 8), (8, 10),
          (5, 11), (6, 12), (11, 12),
          (11, 13), (13, 15), (12, 14),
          (14, 16)
    ]
    limbs = []
    for (start_idx, end_idx) in skeleton:
        start=( person_number)*17 + start_idx
        end= (person_number)*17 + end_idx
        limbs.append((start,end))
    return limbs

def joint_in_time(from_person,to_person):
    joints = []
    for i in range(17):
        joints.append((from_person*17 + i, to_person*17 + i))
    return joints

def get_features_graph_from_yolo_results(results_list):
    frames_dict={} # frame_number: [person_id, person_id,...]
    person_dict={} # person_id: {'joints':..., 'confidence':...}
    joints_all_frames = []
    limb_all_frames = []
    joints_in_time=[]
    current_id=-1
    for time, results in enumerate(results_list):
      frames_dict[time]=[]
      if len(results[0])==0 :
        continue
      distances = []
      for person_id, person in enumerate(results[0]):
        current_id+=1

        joints = person.keypoints.xy.cpu().numpy()[0]
        confidence = person.keypoints.conf.cpu().numpy()[0]
        limbs = get_limbs_person(current_id)
        person_dict[current_id]={
            'joints': joints,
            'confidence': confidence,
            'limbs': limbs
        }

        for idx, joint in enumerate(joints):
          joints_all_frames.append((joint[0], joint[1], confidence[idx]))
        for limb in person_dict[current_id]['limbs']:
          limb_all_frames.append(limb)  
    
        frames_dict[time].append(current_id)

        # Link to previous frame if exits
        if(frames_dict.get(time-1) is None):
          # print(f'No previous frame to link person {current_id} (frame {time})')

          continue
        # Compute distances to all persons in previous frame
        for person_past_frame in frames_dict[time-1]:
          dist = np.linalg.norm(person_dict[current_id]['joints']-person_dict[person_past_frame]['joints'])
          distances.append((dist, person_past_frame, current_id))
      
      # Connect persons based on minimum distance
      # print(f'Computed distances for frame {time}: {distances}')
      while(len(distances)>0):
        distances = sorted(distances, key=lambda x: x[0])
        dist, past_id, current_id = distances.pop(0)
        if dist < 120:
          # print(f'Linking person {past_id} (frame {time-1}) to person {current_id} (frame {time}) with distance {dist}')
          joints_in_time+= joint_in_time(past_id, current_id)


        distances = [d for d in distances if d[1]!=past_id and d[2]!=current_id]
    return joints_all_frames, limb_all_frames, joints_in_time

def get_features(path_video,frame_count=10):
    frames = get_frames(path_video,frame_count)
    results_list = [modelYolo(frame, verbose=False) for frame in frames]
    return get_features_graph_from_yolo_results(results_list)

def build_graph(joints_all_frames, limb_all_frames, joints_in_time, label):
    # Ensure label is wrapped for rank 1 (assuming fix from previous steps)
    if not isinstance(label, (list, tuple)):
        label = [label] 

    # --- FIX FOR EMPTY NODES ---
    num_joints = len(joints_all_frames)
    if num_joints == 0:
        # Create an explicit empty tensor with the expected (0, 3) shape
        joint_features_tensor = tf.zeros((0, 3), dtype=tf.float32) 
    else:
        joint_features_tensor = tf.constant(joints_all_frames, dtype=tf.float32)

    # --- FIX FOR EMPTY LIMB EDGES ---
    num_limbs = len(limb_all_frames)
    if num_limbs == 0:
        limb_sources_tensor = tf.constant([], dtype=tf.int32)
        limb_targets_tensor = tf.constant([], dtype=tf.int32)
    else:
        limb_sources_tensor = tf.constant([limb[0] for limb in limb_all_frames], dtype=tf.int32)
        limb_targets_tensor = tf.constant([limb[1] for limb in limb_all_frames], dtype=tf.int32)

    # --- FIX FOR EMPTY TEMPORAL EDGES ---
    num_temporal = len(joints_in_time)
    if num_temporal == 0:
        temporal_sources_tensor = tf.constant([], dtype=tf.int32)
        temporal_targets_tensor = tf.constant([], dtype=tf.int32)
    else:
        temporal_sources_tensor = tf.constant([joint[0] for joint in joints_in_time], dtype=tf.int32)
        temporal_targets_tensor = tf.constant([joint[1] for joint in joints_in_time], dtype=tf.int32)


    # Now construct the GraphTensor with the fixed tensors
    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={'label': tf.constant(label, dtype=tf.int32)},
        ),

        node_sets={
            "joints": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([num_joints], dtype=tf.int32),
                features={
                    # Use the correct feature name (tfgnn.HIDDEN_STATE)
                    tfgnn.HIDDEN_STATE: joint_features_tensor 
                }
            )
        },
        edge_sets={
            "limbs": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([num_limbs], dtype=tf.int32),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("joints", limb_sources_tensor),
                    target=("joints", limb_targets_tensor)
                )
            ),
            "temporal_connections": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([num_temporal], dtype=tf.int32),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("joints", temporal_sources_tensor),
                    target=("joints", temporal_targets_tensor)
                )
            )
        }
    )
    return graph

def graph_from_video(path_video,name,n_frames=10):
    joints_all_frames, limb_all_frames, joints_in_time = get_features(path_video,n_frames)
    label = get_class_ids(name)
    graph = build_graph(joints_all_frames, limb_all_frames, joints_in_time,label)
  
    return graph


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