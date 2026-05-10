import os
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gradio as gr
import tempfile
import imageio
from pathlib import Path

# Import from stream_inference
from stream_inference import process_video, StreamInference
from train import get_dataset
from model_config import MODELS_CONFIG

# Dataset paths
DATASET_DIR = "Real Life Violence Dataset"
VIDEO_FILE_TYPES = [".mp4", ".avi", ".mov", ".mkv"]

def get_video_options():
    """Get test-set videos from the dataset."""
    videos = {}
    try:
        dataset = get_dataset(DATASET_DIR)
        for file_path, _ in dataset.get('test', []):
            rel_path = os.path.relpath(file_path, DATASET_DIR).replace('\\', '/')
            videos[rel_path] = file_path
    except Exception as e:
        print(f"Warning: could not load dataset test split: {e}")
    return dict(sorted(videos.items()))

def get_model_names():
    """Get all available model names"""
    return [model["name"] for model in MODELS_CONFIG]


def resolve_uploaded_video_path(uploaded_video):
    if not uploaded_video:
        return None
    if isinstance(uploaded_video, dict):
        return uploaded_video.get("tmp_path") or uploaded_video.get("name")
    if isinstance(uploaded_video, str):
        return uploaded_video
    return getattr(uploaded_video, "name", None)


def stream_inference_handler(frame, model_name, state):
    if state is None:
        state = StreamInference()
    processed_frame = state.process_frame(frame, model_name)
    return processed_frame, state
def process_and_display(model_name, video_choice, uploaded_video=None):
    """Process video with selected model or uploaded file and return output"""
    if not model_name:
        return None, "Please select a model"

    video_path = None
    source_label = None

    if uploaded_video:
        uploaded_path = resolve_uploaded_video_path(uploaded_video)
        if uploaded_path and os.path.exists(uploaded_path):
            video_path = uploaded_path
            source_label = "uploaded video"

    if not video_path and video_choice:
        video_options = get_video_options()
        video_path = video_options.get(video_choice)
        source_label = video_choice

    if not video_path:
        return None, "Please select a test-set video or upload your own video."

    if not os.path.exists(video_path):
        return None, f"Video file not found: {video_path}"

    try:
        config = next((m for m in MODELS_CONFIG if m["name"] == model_name), None)
        if not config:
            return None, f"Model '{model_name}' not found"

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name

        print(f"Processing {source_label} with model: {model_name}")
        process_video(video_path, output_path, config)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, "Error: Processed video is empty or was not created."

        web_output_path = output_path.replace(".mp4", "_web.mp4")
        print("Re-encoding video for web compatibility...")
        try:
            reader = imageio.get_reader(output_path)
            fps = reader.get_meta_data().get('fps', 30)
            writer = imageio.get_writer(web_output_path, fps=fps, codec='libx264', quality=8)
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            return web_output_path, f"✓ Successfully processed {source_label} with {model_name}"
        except Exception as re_enc_error:
            print(f"Re-encoding failed: {re_enc_error}")
            return output_path, f"✓ Processed {source_label} with {model_name} (Re-encoding failed, video may not display)"

    except Exception as e:
        return None, f"Error: {str(e)}"

def create_gradio_interface():
    
    model_options = get_model_names()
    video_keys = list(get_video_options().keys())
    
    # Create the interface
    with gr.Blocks(title="Violence Detection Inference") as demo:
        gr.Markdown("# 🎥 Violence Detection Inference")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=model_options,
                label="Select Model",
                value=model_options[0] if model_options else None,
                info="Choose a trained model",
                scale=1
            )
            
            video_dropdown = gr.Dropdown(
                choices=video_keys,
                label="Select Test-Set Video",
                value=video_keys[0] if video_keys else None,
                info="Choose a video from the test split",
                scale=1
            )
        
        with gr.Row():
            uploaded_video = gr.File(
                label="Or upload a video",
                file_count="single",
                type="filepath",
                file_types=[".mp4", ".avi", ".mov", ".mkv"]
            )
        
        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready. Select a test video or upload your own, then click 'Process Video'",
                scale=1
            )
            
            process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg", scale=0)
        
        # Output video player
        with gr.Row():
            output_video = gr.Video(
                label="Inference Result",
                interactive=False,
                autoplay=True
            )

        
        ### TODO: change to https://www.gradio.app/guides/object-detection-from-webcam-with-webrtc

        # https://www.gradio.app/guides/streaming-inputs
        # Simple approach without requiring additional libraries
        with gr.Row():
            with gr.Column():
                options = gr.WebcamOptions(
                    mirror=True, 
                    constraints={
                        "width": {"ideal": 128, "max":240},
                        "height": {"ideal": 128,"max":240},
                        "frameRate": {"ideal": 15, "max":15}
                    }
                )
                input_img = gr.Image(label="Input", sources="webcam", type="numpy", webcam_options=options)
            with gr.Column():
                output_img = gr.Image(label="Output")

        # Connect the button click to the processing function
        stream_state = gr.State(None)
        input_img.stream(
            fn=stream_inference_handler,
            inputs=[input_img, model_dropdown, stream_state],
            outputs=[output_img, stream_state],
            time_limit=30,
            stream_every=0.5,
            concurrency_limit=2
        )
        process_btn.click(
            fn=process_and_display,
            inputs=[model_dropdown, video_dropdown, uploaded_video],
            outputs=[output_video, status_text]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting Gradio app...")
    print(f"Available models: {len(get_model_names())}")
    print(f"Available videos: {len(get_video_options())}")
    
    demo = create_gradio_interface()
    demo.launch( server_name="0.0.0.0", server_port=7860)
