# Computer Vision Skeleton Analysis for Violence Recognition

This project implements various pose-based deep learning models to detect violence in video sequences. It utilizes human skeleton keypoints extracted via YOLOv11-pose and processes them using three main architectures ST-GCN, SPIL, and PoseConv3D.

## Features

- **Multi-Model Support**: Includes ST-GCN (Spatial-Temporal Graph Convolutional Networks), SPIL (Skeleton-based Point-set Interaction Learning), and PoseConv3D.
- **Keypoint Extraction**: Automated human pose estimation using YOLOv11.
- **Stream Inference**: Processing of video files with annotated overlays.
- **Benchmarking**: Performance analysis of different architectures.
- **Interactive Web App**: User-friendly Gradio interface for easy testing.

---

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Weights**: Ensure you have `yolo11n-pose.pt` in the root directory and trained model weights in the `models/` folder.

---

## Training

You can train a model using the `train.py` script. You can either specify a configuration from `model_config.py` or define parameters manually.

**Using a config:**

```bash
python train.py --config SPIL-f10 --epochs 50 --batch_size 16
```

**Manual parameters:**

```bash
python train.py --model SPIL --frames 15 --epochs 100
```

### Arguments:

- `--config`: Name of the model configuration in `model_config.py`.
- `--model`: Model type (`POSE_CONV3D`, `ST_CGN`, `SPIL`).
- `--epochs`: Number of training epochs (default: 100).
- `--batch_size`: Batch size (default: 8).
- `--frames`: Number of frames to sample per video.
- `--dataset_path`: Path to the dataset (default: `./Real Life Violence Dataset`).
- `--resume_id`: Weights & Biases run ID to resume training.

---

## Benchmarking

Evaluate the performance (inference time and preprocessing speed) of your models using `benchmark.py`.

**Benchmark a specific model:**

```bash
python benchmark.py --model SPIL-f10
```

**Benchmark all models in config:**

```bash
python benchmark.py --all
```

Results are automatically saved to `benchmark_results.csv`.

---

## Stream Inference

Process a video file and save the output with violence detection annotations.

```bash
python stream_inference.py --input path/to/video.mp4 --output results/output.mp4 --model SPIL-f10
```

### Arguments:

- `--input`: Path to the input video file.
- `--output`: Path where the annotated video will be saved.
- `--model`: Model name from `model_config.py`.

---

## Gradio App

Launch an interactive web interface to test the models on the dataset videos.

```bash
python gradio_app.py
```

Once launched, the app will be available at `http://localhost:7860`. It allows you to:

1. Select a trained model.
2. Select a video from the dataset.
3. View the real-time inference result in the browser.

---

## Downloading the Dataset

If you don't have the dataset, you can download it using the Kaggle API. Ensure you have your `kaggle.json` file in the project root:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
unzip real-life-violence-situations-dataset.zip -d "Real Life Violence Dataset"
```

## Dataset Structure

The project expects the following dataset structure:

```
Real Life Violence Dataset/
├── Violence/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── NonViolence/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

---

## Architecture & Scaffolding

This project follows a structure where each action recognition architecture (STCGN, SPIL, PoseConv3D) is contained in its own directory with a consistent internal scaffolding:

```
[Module_Name]/
├── generator.py          # Handles data iteration and batching
├── feature_extractor.py  # Transforms YOLO keypoints to model input
└── model.py              # The deep learning architecture (TF/Keras)
```

### Component Roles

1.  **Generator**: Responsible for iterating through the dataset, managing batching, and providing samples to the training process. The scope to use data generators allow to reduce memory usage by mapping on the fly the data from the disk to the memory. In this way we can train models with a large number of frames without loading all the data into the memory.

2.  **Feature Extractor**: Orchestrates the preprocessing of raw video frames. It uses YOLOv11-pose to detect human keypoints and transforms them into the specific format required by the model (graphs, point clouds, or heatmaps).

3.  **Model**: The deep learning architecture that performs the action classification based on the extracted features.

---

### Implemented Models

For detailed technical descriptions of each architecture, please refer to their respective documentation:

- **[STCGN (Spatial-Temporal Graph Convolutional Network)](./STCGN/README.md)**: Processes skeleton data as a graph
- **[SPIL (Skeleton-based Point-set Interaction Learning)](./SPIL/README.md)**: Treats skeletons as a 3D point cloud with dynamic interaction modeling.

- **[PoseConv3D](./PoseConv3D/README.md)**: Represents skeleton data as 3D heatmaps for processing with 3D CNNs.

---

## Core Utilities

### Benchmarking (`benchmark.py`)

Provides a suite to evaluate model performance. It measures:

- **Inference Time**: Time taken for the model to produce a prediction.
- **Preprocessing Time**: Time taken to extract features from raw video.

### Stream Processing (`stream_inference.py`)

Handles real-time or file-based video processing. It orchestrates the entire pipeline: reading frames, extracting features via YOLO, transforming the keypoints to model inputs, running the classification model, and overlaying the results (e.g., "Violence" vs "Non-Violence" labels) on the output video the result is a new video with the annotations of the skeleton keypoints detected (yolo) and the prediction of the model.

### Loader (`loader.py`)

A centralized utility to manage the loader model from the las best model implementation trained . it builds the model base on the model_config.py file and loads the weights from the trained model.

### Dataset (`dataset.py`)

A centralized utility to manage the dataset . it builds the Train/Validation/Test split that are fit into the datagenerators, also use in the benchmarking process and the gradio app to allow select only test samples for evaluation.

It save the splits into a json file to be reused and avoid shuffling the dataset every time also to warranty the same samples for training and validation for all the models.
