
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