from PIL import Image
import os

def is_black_image(image_path, threshold=10):
    image = Image.open(image_path).convert("L")
    pixel_avg = sum(image.getdata()) / len(image.getdata())
    return pixel_avg < threshold

root_dir = "cifar10"
black_images = []
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        if is_black_image(image_path):
            black_images.append(image_path)

print("黑色图片路径列表：", black_images)
