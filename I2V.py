import torch
from diffusers import I2VGenXLPipeline
import cv2
import numpy as np
import os
from PIL import Image

def create_video(image_path, video_prompt, output_dir="generated_videos", video_filename="generated_video.mp4"):
    """
    Generate a video using the I2VGenXLPipeline with the given image and video prompt.
    Saves the generated video in the specified output directory as an MP4 file.

    Args:
        image_path (str): Path to the input image.
        video_prompt (str): Prompt for video generation.
        output_dir (str): Directory where the generated video will be saved.
        video_filename (str): Name of the generated video file (MP4 format).

    Returns:
        str: Path to the generated video.
    """
    try:
        # Define model cache directory
        model_cache_dir = "models"

        # Load the I2VGenXL pipeline and set to GPU mode (optimized for H100)
        pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl",
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_cache_dir  # Specify the cache directory
        )

        # Load the locally saved image
        image = Image.open(image_path).convert("RGB")

        # Set negative prompt (optional)
        negative_prompt = (
            "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, "
            "static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        )

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate video frames
        generator = torch.manual_seed(8888)
        frames = pipeline(
            prompt=video_prompt,
            image=image,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
        ).frames[0]

        # Prepare the frames for video export (convert frames to the correct format)
        frames = [np.array(frame) for frame in frames]
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]  # Convert RGB to BGR

        # Define the video file path
        video_path = os.path.join(output_dir, video_filename)

        # Create a VideoWriter object using OpenCV to save the frames as an MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        fps = 30  # Frames per second
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write frames to the video file
        for frame in frames:
            video_writer.write(frame)

        # Release the video writer object and save the video
        video_writer.release()

        print(f"Video saved at: {video_path}")
        return video_path

    except Exception as e:
        print(f"Error generating video: {e}")
        return None
