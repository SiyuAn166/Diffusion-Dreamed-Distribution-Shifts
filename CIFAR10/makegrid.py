import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

airplane_data = []

for i in range(2):
    for c in class_names:
        img = Image.open(os.path.join(f'cifar10_32/{c}', f'{c}_{i}.png'))
        img = transform(img)
        airplane_data.append(img)
vutils.save_image(airplane_data, 'merged.png', nrow=10, normalize=True)
