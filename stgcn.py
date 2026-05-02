
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow import keras
import random
from preprocessing import get_frames,get_class_ids
import numpy as np

from ultralytics import YOLO
modelYolo = YOLO("yolo11n-pose.pt")


# Utilizaremos la especificación del grafo que definimos previamente
FRAME_COUNT =10
HEIGHT= 128
WIDTH = 128
CHANNELS = 17
FRAME_COUNT =10

def create_skeleton_graph_spec_with_label():
    """Define la estructura del Grafo Espacio-Temporal con label (para la entrada del modelo)."""
    # Nota: El modelo solo necesita la estructura de entrada, no la etiqueta en el Contexto.
    # Pero aquí definimos la estructura completa por conveniencia.
    return tfgnn.GraphTensorSpec.from_piece_specs(
        context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
            # Definimos solo la estructura mínima de contexto para la entrada del modelo
            'label': tf.TensorSpec(shape=(1,), dtype=tf.int32)
        }),
        node_sets_spec={
            'joints': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 3), tf.float32)},
                sizes_spec=tf.TensorSpec((1,), tf.int32))
        },
        edge_sets_spec={
            'limbs': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={}, sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets('joints', 'joints')),
            'temporal_connections': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={}, sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets('joints', 'joints'))
        }
    )
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
def dense(units, activation="relu"):
    """A Dense layer with regularization (L2 and Dropout)."""
    l2_regularization=5e-4
    dropout_rate=0.1
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout_rate)
    ])

def ST_GCN(graph_spec, gnn_units=64, num_gcn_layers=2):
    """
    Crea y compila un modelo GNN basado en GCN para la clasificación de grafos.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): La especificación de la entrada del grafo.
        gnn_units (int): La dimensión de la incrustación oculta para los nodos.
        num_gcn_layers (int): Número de capas GCN a aplicar.
        
    Returns:
        tf.keras.Model: El modelo Keras compilado.
    """
    # 1. ENTRADA
    graph_input = keras.Input(type_spec=graph_spec)
    graph = graph_input.merge_batch_to_components()
    message_dim = 16
    next_state_dim=32


    # 2. PROCESAMIENTO INICIAL (Si se desea un embedding inicial)
    # No es necesario aquí, ya que las features [x, y, conf] ya son útiles.
    
    # 3. CAPAS GNN (Propagación de Mensajes Espacio-Temporal)
    # Aplicar la GCN a través de los diferentes tipos de bordes
    for i in range(num_gcn_layers):
        
        # Propagación de mensajes *Espaciales* ('limbs')
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "joints": tfgnn.keras.layers.NodeSetUpdate({
                    "limbs":tfgnn.keras.layers.SimpleConv(
                        sender_node_feature=tfgnn.HIDDEN_STATE,
                        message_fn=dense(message_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET
                    )
                },
                tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim))),
            })(graph)
        
        # Propagación de mensajes *Temporales* ('temporal_connections')
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "joints": tfgnn.keras.layers.NodeSetUpdate({
                    "temporal_connections": tfgnn.keras.layers.SimpleConv(
                        sender_node_feature=tfgnn.HIDDEN_STATE,
                        message_fn=dense(message_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET
                    )
                },
                tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)))
            })(graph)
    
    
    # 4. AGREGACIÓN GLOBAL (De Nodos a Grafo)
    # Combinar todas las features de los nodos 'joints' para obtener una feature única para el grafo.
    graph_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="joints"
    )(graph)
    
    
    # 5. CLASIFICACIÓN (MLP para el Contexto)
    
    # Capa Densa (Regularización)
    classification_output = keras.layers.Dense(32, activation='relu')(graph_features)
    classification_output = keras.layers.Dropout(0.1)(classification_output)
    
    # Capa de Salida (1 unidad para clasificación binaria)
    output = keras.layers.Dense(2,  activation='softmax')(classification_output)
    
    # Definición del modelo
    model = keras.Model(inputs=graph_input, outputs=output)
    
    # Compilación del modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model
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