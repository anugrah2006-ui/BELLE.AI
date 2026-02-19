import asyncio
import os
import requests
from random import randint
from PIL import Image
from dotenv import get_key
from time import sleep

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(script_dir, "..", ".env"))

def open_images(prompt):
    folder_path = os.path.join(script_dir, "Data")
    prompt_clean = prompt.replace(" ", "_")

    for i in range(1, 5):
        jpg_file = f"{prompt_clean}_{i}.jpg"
        image_path = os.path.join(folder_path, jpg_file)
        try:
            img = Image.open(image_path)
            print(f"Opening image: {image_path}")
            img.show()
            sleep(1) # Small delay to let the default viewer open
        except IOError:
            print(f"Error opening image: {image_path}")

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

# Using absolute path for .env file
hf_token = get_key(env_path, "Hugging_Face_api")
Header = {"Authorization": f"Bearer {hf_token}"}

async def query(payload):
    response = requests.post(API_URL, headers=Header, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        print(f"[ERROR] API returned status code {response.status_code}: {response.text}")
        return None

async def generate_images(prompt: str):
    tasks = []
    prompt_clean = prompt.replace(' ', '_')
    data_dir = os.path.join(script_dir, "Data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in range(4):
        payload = {
            "inputs": (
                f"{prompt}, quality=4K, sharpness=maximum, "
                f"Ultra High details, high resolution, "
                f"seed={randint(0, 1_000_000)}"
            )
        }
        task = asyncio.create_task(query(payload))
        tasks.append(task)

    image_bytes_list = await asyncio.gather(*tasks)

    for i, image_bytes in enumerate(image_bytes_list):
        if image_bytes:
            file_path = os.path.join(data_dir, f"{prompt_clean}_{i + 1}.jpg")
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            print(f"[SUCCESS] Saved: {file_path}")

def GenerateImages(prompt: str):
    asyncio.run(generate_images(prompt))
    open_images(prompt)

# Correct path for the data file used by front-end
data_file_path = os.path.abspath(os.path.join(script_dir, "..", "Frontend", "Files", "ImageGeneration.data"))

if __name__ == "__main__":
    print(f"[INFO] Monitoring {data_file_path}...")
    while True:
        try:
            if os.path.exists(data_file_path):
                with open(data_file_path, "r") as f:
                    content = f.read().strip()
                
                if content:
                    parts = content.split(",")
                    if len(parts) == 2:
                        prompt, status = parts
                        
                        if status.strip().lower() == "true":
                            print(f"[PROCESS] Generating images for: {prompt}")
                            GenerateImages(prompt=prompt.strip())

                            with open(data_file_path, "w") as f:
                                f.write("False,False")
                            print("[INFO] Image generation complete and status reset.")
                            # break # Optional: remove if you want it to keep running
                    
            sleep(1)
        except Exception as e:
            print(f"[ERROR] {e}")
            sleep(1)
