import os
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gradio as gr
import tempfile
import imageio
from pathlib import Path

# Import from stream_inference
from stream_inference import process_video
from model_config import MODELS_CONFIG

# Dataset paths
DATASET_DIR = "Real Life Violence Dataset"
VIOLENCE_DIR = os.path.join(DATASET_DIR, "Violence")
NON_VIOLENCE_DIR = os.path.join(DATASET_DIR, "NonViolence")

def get_video_options():
    """Get all available videos from the dataset"""
    videos = {}
    
    # Get violence videos
    if os.path.exists(VIOLENCE_DIR):
        violence_videos = sorted([f for f in os.listdir(VIOLENCE_DIR) if f.endswith(('.mp4', '.avi'))])
        for v in violence_videos:
            videos[f"Violence/{v}"] = os.path.join(VIOLENCE_DIR, v)
    
    # Get non-violence videos
    if os.path.exists(NON_VIOLENCE_DIR):
        non_violence_videos = sorted([f for f in os.listdir(NON_VIOLENCE_DIR) if f.endswith(('.mp4', '.avi'))])
        for nv in non_violence_videos:
            videos[f"NonViolence/{nv}"] = os.path.join(NON_VIOLENCE_DIR, nv)
    
    return videos

def get_model_names():
    """Get all available model names"""
    return [model["name"] for model in MODELS_CONFIG]

def process_and_display(model_name, video_choice):
    """Process video with selected model and return output"""
    
    if not model_name or not video_choice:
        return None, "Please select both a model and a video"
    
    try:
        # Find the model config
        config = next((m for m in MODELS_CONFIG if m["name"] == model_name), None)
        if not config:
            return None, f"Model '{model_name}' not found"
        
        # Get video path
        video_options = get_video_options()
        video_path = video_options.get(video_choice)
        
        if not video_path or not os.path.exists(video_path):
            return None, f"Video file not found: {video_choice}"
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Process the video
        print(f"Processing video: {video_choice} with model: {model_name}")
        process_video(video_path, output_path, config)
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return None, f"Error: Processed video is empty or was not created."
            
        # Return the output video path
        # Re-encode for web compatibility if it's not playing
        print("Re-encoding video for web compatibility...")
        web_output_path = output_path.replace(".mp4", "_web.mp4")
        try:
            reader = imageio.get_reader(output_path)
            fps = reader.get_meta_data().get('fps', 30)
            writer = imageio.get_writer(web_output_path, fps=fps, codec='libx264', quality=8)
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            return web_output_path, f"✓ Successfully processed with {model_name}"
        except Exception as re_enc_error:
            print(f"Re-encoding failed: {re_enc_error}")
            return output_path, f"✓ Processed with {model_name} (Re-encoding failed, video may not display)"
        
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
                label="Select Video",
                value=video_keys[0] if video_keys else None,
                info="Choose a video from the dataset",
                scale=1
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready. Select model and video, then click 'Process Video'",
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
        # Connect the button click to the processing function
        process_btn.click(
            fn=process_and_display,
            inputs=[model_dropdown, video_dropdown],
            outputs=[output_video, status_text]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting Gradio app...")
    print(f"Available models: {len(get_model_names())}")
    print(f"Available videos: {len(get_video_options())}")
    
    demo = create_gradio_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
