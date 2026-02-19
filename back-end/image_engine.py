from diffusers import StableDiffusionPipeline
import torch

# Load model (first time takes time)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")   # because you have i3

def generate_image(prompt):
    image = pipe(
        prompt,
        num_inference_steps=20,
        height=384,
        width=384
    ).images[0]

    image.save("output.png")
    return "output.png"
