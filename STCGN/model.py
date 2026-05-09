
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
    """Defines the structure of the Spatio-Temporal Graph with label (for model input)."""
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
    Creates a ST-GCN model for skeleton-based action recognition.
    Crea y compila un modelo GNN basado en GCN para la clasificación de grafos.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The specification of the input graph.
        gnn_units (int): The dimension of the hidden embedding for the nodes.
        num_gcn_layers (int): Number of GCN layers to apply.
        
    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # 1. INPUT
    graph_input = keras.Input(type_spec=graph_spec)
    graph = graph_input.merge_batch_to_components()
    message_dim = 16
    next_state_dim=32


    # 2. INITIAL PROCESSING (If an initial embedding is desired)
    # Not necessary here, since the [x, y, conf] features are already useful.
    
    # 3. GNN LAYERS (Spatio-Temporal Message Propagation)
    # Apply the GCN across the different types of edges
    for i in range(num_gcn_layers):
        
        # Spatial message passing ('limbs')
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
        
        # Temporal message passing ('temporal_connections')
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
    
    
    # 4. GLOBAL AGGREGATION (From Nodes to Graph)
    # Combine all node features to obtain a unique feature for the graph.
    graph_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="joints"
    )(graph)
    
    
    # 5. CLASSIFICATION (MLP for the Context)
    
    # Dense Layer (Regularization)
    classification_output = keras.layers.Dense(32, activation='relu')(graph_features)
    classification_output = keras.layers.Dropout(0.1)(classification_output)
    
    # Output Layer (1 unit for binary classification)
    output = keras.layers.Dense(2,  activation='softmax')(classification_output)
    
    # Model definition
    model = keras.Model(inputs=graph_input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model