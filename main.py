import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])

# Define the paths to store the train and test datasets
path_to_store_trainset = './data_train'
path_to_store_testset = './data_test'

# Download and load the MNIST datasets
trainset = datasets.MNIST(path_to_store_trainset, download=True, train=True, transform=transform)
valset = datasets.MNIST(path_to_store_testset, download=True, train=False, transform=transform)

# Create data loaders for training and validation sets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Get an iterator for the training loader
dataiter = iter(trainloader)

# Get the next batch of data
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

input_size = 784
hidden_sizes = [512, 256, 128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], output_size),
    nn.LogSoftmax(dim=1)
).to(device)  # Move the model to the GPU if available

print(model)

criterion = nn.NLLLoss().to(device)  # Move the loss to the GPU if available
images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 12
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)

# Predict on a single image from the validation set
images, labels = next(iter(valloader))
img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img.to(device))  # Move the image to the GPU if available

ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])  # Move probabilities back to the CPU
predicted_digit = probab.index(max(probab))
print("Predicted Digit =", predicted_digit)

correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        with torch.no_grad():
            logps = model(img.to(device))  # Move the image to the GPU if available

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])  # Move probabilities back to the CPU
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))

# Save the model to a file (note that you should specify a valid path)
torch.save(model.state_dict(), 'path_to_save_model.pth')
