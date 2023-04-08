import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score

# Set the path of the folder containing the images to classify
test_data_path = "../val"

# Set the path of the trained model
model_path = "resnet_model_epoch_25.pth"

# Set the number of classes
num_classes = 10

# Set the device to use for testing (GPU if available, CPU otherwise)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations to apply to the images
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                       std=[0.229, 0.224, 0.225])
                          ])

# Load the dataset to classify
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)

# Create the data loader for testing
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Load the trained model
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Test the model
y_true = []
y_pred = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()

# Calculate the confusion matrix and accuracy score
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

for i in range(num_classes):
    cm_class = confusion_matrix(y_true, y_pred, labels=[i])
    acc_class = accuracy_score(y_true, y_pred, normalize=True)
    print("Class:", i)
    print("Confusion matrix:")
    print(cm_class)
    print("Accuracy score:", acc_class)

print("Overall confusion matrix:")
print(cm)
print("Overall accuracy score:", acc)



# import torch
# import torchvision
# from torchvision import transforms
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# import numpy as np
# from dataset import get_data_loader
# # import matplotlib.pyplot as plt

# # transform_test = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # ])
# # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# # testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
# testloader = get_data_loader(batch_size=128, num_workers=0)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = torchvision.models.resnet18()
# net.fc = torch.nn.Linear(net.fc.in_features, 10)
# net.load_state_dict(torch.load('./resnet18.pth'))
# net.to(device)
# net.eval()

# y_pred = []
# y_true = []
# with torch.no_grad():
#     for i, data in enumerate(testloader):
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         y_pred.extend(predicted.cpu().numpy())
#         y_true.extend(labels.cpu().numpy())

# cm = confusion_matrix(y_true, y_pred)
# print(cm)

# accuracy = accuracy_score(y_true, y_pred)
# print("Accuracy:", accuracy)
# # Define class names
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # Plot confusion matrix
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=90)
# plt.yticks(tick_marks, classes)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.tight_layout()

# # Add values to the plot
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")

# plt.show()