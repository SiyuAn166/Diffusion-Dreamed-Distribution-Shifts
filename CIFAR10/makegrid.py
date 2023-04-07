import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

# 用于转换图像的预处理函数
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图像标准化为 [-1, 1]
])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 从 cifar10 文件夹中加载前十张“飞机”图像
airplane_data = []

for i in range(2):
    for c in class_names:
        img = Image.open(os.path.join(f'cifar10_32/{c}', f'{c}_{i}.png'))
        img = transform(img)
        airplane_data.append(img)

# 将图像转换为网格并保存到磁盘
vutils.save_image(airplane_data, 'merged.png', nrow=10, normalize=True)
