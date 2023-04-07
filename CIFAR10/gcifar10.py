import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
import huggingface_hub

# pip install cryptography==38.0.4
def main(prompt):
    huggingface_hub.login(token="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")

    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16"
    )

    pipeline = pipeline.to("cuda")

    folder_path = f"cifar10/{prompt}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(1000):
        images = pipeline(prompt, num_images_per_prompt=1, guidance_scale=8)
        image = images.images[0]
        image_name = f"cifar10/{prompt}/image_{i}.png"
        image.save(image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from a prompt using Stable Diffusion.")
    parser.add_argument("prompt", type=str, help="The prompt used to generate the image.")
    args = parser.parse_args()

    main(args.prompt)
