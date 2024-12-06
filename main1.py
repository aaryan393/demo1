import json
import os
from main import generate_prompts, generate_image  # Import functions from the first file
from I2V import create_video  # Import the function from the second file

def main():
    # Step 1: Get user input for prompt generation
    user_input = input("Enter a description for generating image and video prompts: ")
    
    # Step 2: Generate image and video prompts
    print("Generating prompts...")
    image_prompt, video_prompt = generate_prompts(user_input)
    
    # Save prompts to a local JSON file for debugging
    prompts_file = "prompts.json"
    with open(prompts_file, "w") as f:
        json.dump({"image_prompt": image_prompt, "video_prompt": video_prompt}, f, indent=4)
    print(f"Prompts saved to {prompts_file}")
    
    # Step 3: Generate image using the image prompt
    print("Generating image...")
    image_path = generate_image(image_prompt)
    print(f"Image saved at: {image_path}")
    
    # Save the generated image path and prompts together for reference
    output_metadata = {
        "image_prompt": image_prompt,
        "video_prompt": video_prompt,
        "image_path": image_path
    }
    metadata_file = "output_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(output_metadata, f, indent=4)
    print(f"Metadata saved to {metadata_file}")
    
    # Step 4: Generate video using the saved image and video prompt
    print("Generating video...")
    video_path = create_video(image_path, video_prompt)
    print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    main()
