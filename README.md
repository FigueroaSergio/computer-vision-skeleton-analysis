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

## Experiment Tracking

This project uses **Weights & Biases (WandB)** for experiment tracking. Make sure to log in before training:

```bash
wandb login
```
