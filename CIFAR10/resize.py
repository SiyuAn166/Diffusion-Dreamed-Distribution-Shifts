from PIL import Image
import os

input_dir = "cifar10"
output_dir = "cifar10_32"
target_size = (32, 32)

for class_dirname in os.listdir(input_dir):
    class_dirpath = os.path.join(input_dir, class_dirname)
    if os.path.isdir(class_dirpath):
        for filename in os.listdir(class_dirpath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                
                image = Image.open(os.path.join(class_dirpath, filename))
                resized_image = image.resize(target_size)
                output_class_dirpath = os.path.join(output_dir, class_dirname)
                if not os.path.exists(output_class_dirpath):
                    os.makedirs(output_class_dirpath)
                output_filepath = os.path.join(output_class_dirpath, filename)
                resized_image.save(output_filepath)
                print(f"save image path: {output_filepath}")
