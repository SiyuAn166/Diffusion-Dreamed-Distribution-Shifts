import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Set the path of the folder containing the subfolders of images
data_path = "../images"

# Set the number of classes
num_classes = 10

# Set the size of the batch
batch_size = 64

# Set the number of epochs
num_epochs = 35

# Set the device to use for training (GPU if available, CPU otherwise)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations to apply to the images
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                       std=[0.229, 0.224, 0.225])
                          ])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Define the ResNet model
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # Save the model every 5 epochs
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), 'resnet_model_epoch_{}.pth'.format(epoch+1))
print('Finished training')

# # Save the model
# torch.save(model.state_dict(), "resnet_model.pth")


# import torch
# import pickle
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# from dataset import get_data_loader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)  
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # transform = transforms.Compose([
# #     transforms.RandomHorizontalFlip(),
# #     transforms.RandomCrop(32, padding=4),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# # ])

# # trainset = Dataset(train=True, transform=transform, custom_images_path='cifar10_32')

# # trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
# trainloader = get_data_loader(batch_size=32, num_workers=0)


# for epoch in range(50):  
#     running_loss = 0.0
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 200 == 199:    
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0

# print('Finished Training')

# PATH = './resnet18.pth'
# torch.save(model.state_dict(), PATH)