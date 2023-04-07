# import os
# import torch
# from diffusers import StableDiffusionPipeline
# import huggingface_hub

# black_images = ['cifar10/deer/image_400.png', 'cifar10/deer/image_526.png', 'cifar10/deer/image_935.png', 'cifar10/horse/image_60.png']
# prompts = [item.split('/')[1] for item in black_images]

# huggingface_hub.login(token="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")

# pipeline = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16"
# )

# pipeline = pipeline.to("cuda")

# for prompt, path in zip(prompts, black_images):
#     images = pipeline(prompt, num_images_per_prompt=1, guidance_scale=8)
#     image = images.images[0]
#     image_name = path
#     if sum(image.convert("L").getdata()) / len(image.convert("L").getdata()) > 10:
#         image.save(image_name)
#         print(f"image saved for {path}")

from PIL import Image
import os

# 设置输入输出目录和目标分辨率
input_dir = "cifar10"
output_dir = "cifar10_32"
target_size = (32, 32)

# 遍历输入目录下的每个类别文件夹
for class_dirname in os.listdir(input_dir):
    class_dirpath = os.path.join(input_dir, class_dirname)
    if os.path.isdir(class_dirpath):
        # 如果当前路径是目录，遍历其中的所有图片文件
        for filename in os.listdir(class_dirpath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 读取图片
                image = Image.open(os.path.join(class_dirpath, filename))

                # 缩放图片
                resized_image = image.resize(target_size)

                # 创建输出目录，并保存缩放后的图片
                output_class_dirpath = os.path.join(output_dir, class_dirname)
                if not os.path.exists(output_class_dirpath):
                    os.makedirs(output_class_dirpath)
                output_filepath = os.path.join(output_class_dirpath, filename)
                resized_image.save(output_filepath)
                print(f"save image path: {output_filepath}")
