import os
import requests
from PIL import Image
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
#from azure.ai.openai import OpenAIClient as AzureOpenAIClient

# Load environment variables from .env
load_dotenv()

# Configuration settings from .env
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")

AZURE_OPENAI_ENDPOINT = "https://776969847273.openai.azure.com/"  # Your Azure OpenAI Endpoint
AZURE_OPENAI_KEY = "e719b3bb8ced4f4aacc8f70ab96e59e4"  # Your Azure OpenAI Key
DEPLOYMENT_NAME = "gpt-35-turbo-2" 

# Configure client
client1 = AzureOpenAI(
    api_version="2024-02-01",  
    api_key=AZURE_OPENAI_KEY,  
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
# Setup OpenAI client
client2 = AzureOpenAI(
    api_version="2024-02-01",  
    api_key=azure_oai_key,  
    azure_endpoint=azure_oai_endpoint
)

# Function to generate image and video prompts
def generate_prompts(input_text):
    """
    Generate two prompts: one for an image and another for a video.
    """
    try:
        # Generate image prompt
        system_message_image = "Generate a detailed and creative image prompt from the user's input."
        response_image = client1.chat.completions.create(
            model=DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_message_image},
                {"role": "user", "content": input_text}
            ]
        )
        image_prompt = response_image.choices[0].message.content.strip()

        # Generate video prompt
        system_message_video = "Generate a detailed video prompt based on the user's input."
        response_video = client1.chat.completions.create(
            model=DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_message_video},
                {"role": "user", "content": input_text}
            ]
        )
        video_prompt = response_video.choices[0].message.content.strip()

        return image_prompt, video_prompt
    except Exception as e:
        return f"Error generating prompts: {e}"

# Function to generate an image using DALL·E
def generate_image(prompt):
    """
    Generate an image using DALL·E model.
    """
    try:
        response = client2.images.generate(
            model=azure_oai_deployment,  # Specify your DALL·E model name
            prompt=prompt,
            n=1
        )
        json_response = json.loads(response.model_dump_json())
        image_url = json_response["data"][0]["url"]

        # Download and save the image
        image_dir = os.path.join(os.curdir, 'generated_images')
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        image_path = os.path.join(image_dir, "generated_image.png")
        generated_image = requests.get(image_url).content
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)

        # Display the image
        image = Image.open(image_path)
        image.show()

        return image_path
    except Exception as e:
        return f"Error generating image: {e}"

# Main function
if __name__ == "__main__":
    user_input = input("Enter a description for generating image and video prompts: ")

    # Generate prompts using GPT-3.5 Turbo
    image_prompt, video_prompt = generate_prompts(user_input)
    print(f"Image Prompt: {image_prompt}")
    print(f"Video Prompt: {video_prompt}")

    # Generate image from the image prompt using DALL·E
    image_path = generate_image(image_prompt)
    print(f"Generated Image saved at: {image_path}")
