import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./data', train=True, transform=None, custom_images_path=None):
        self.transform = transform
        self.custom_images_path = custom_images_path
        self.trainset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        _data = torch.from_numpy(self.trainset.data).permute(0,3,1,2) # [b, 3, 32, 32]
        _targets = torch.tensor(self.trainset.targets)
        data = []
        targets = []

        self.dataset = None
        self.targets = None

        classes = self.trainset.classes
        if custom_images_path is not None:
            classes = self.trainset.classes
            for i, class_name in enumerate(classes):
                class_path = os.path.join(custom_images_path, class_name)
                if not os.path.exists(class_path):
                    continue
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    img = Image.open(file_path)
                    img = transform(img)
                    data.append(img.numpy())
                    targets.append(i)
        data = torch.tensor(np.array(data))
        targets = torch.tensor(targets)

        self.dataset = torch.cat([_data, data], dim=0)
        self.targets = torch.cat([_targets, targets], dim=0)

        # synthetic
        # self.dataset = data
        # self.targets = targets

    def __getitem__(self, index):
        return self.dataset[index], self.targets[index]

    def __len__(self):
        return len(self.dataset)

 