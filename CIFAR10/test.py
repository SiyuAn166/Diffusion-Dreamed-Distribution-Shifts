import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import Dataset
# import matplotlib.pyplot as plt

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torchvision.models.resnet18()
net.fc = torch.nn.Linear(net.fc.in_features, 10)
net.load_state_dict(torch.load('./resnet18_ori.pth'))
net.to(device)
net.eval()

y_pred = []
y_true = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
print(cm)

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
# # Define class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class_acc = []
for i in range(10):
    acc_i = cm[i,i]/np.sum(cm[i,:])
    class_acc.append(acc_i)
    print(f"Accuracy of class {classes[i]}: {acc_i}")