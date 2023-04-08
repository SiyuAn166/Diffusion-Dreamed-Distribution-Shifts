import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score

# Set the path of the folder containing the images to classify
test_data_path = "../val"

# Set the path of the trained model
model_path = "resnet_model_epoch_35.pth"

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