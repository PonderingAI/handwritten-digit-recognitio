import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time
import hyperparameters_v3 as hyp
from hyperparameters_v3 import neuron, model
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler



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


hyp.neuron
hyp.model 

# Define loss
criterion = nn.NLLLoss()

print(hyp.model)

criterion = nn.NLLLoss().to(device)

optimizer = optim.SGD(hyp.model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15

# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 3  # Number of epochs to wait before early stopping

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Move data to the same device as the hyp.model
        images, labels = images.to(device), labels.to(device)

        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = hyp.model(images.to(device))
        loss = criterion(output, labels)

        # This is where the hyp.model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

    # ... (validation and early stopping code as before)



        # Validate the hyp.model and monitor the validation loss
        val_loss = 0
        hyp.model.eval()  # Set the hyp.model to evaluation mode

        with torch.no_grad():
            for images, labels in valloader:
                images = images.to(device) 
                labels = labels.to(device)  
                images = images.view(images.shape[0], -1)
                output = hyp.model(images)
                val_loss += criterion(output, labels)

        hyp.model.train()  # Set the hyp.model back to training mode

        val_loss = val_loss / len(valloader)
        print("Epoch {} - Validation loss: {}".format(e, val_loss))

        # Check if validation loss is decreasing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best hyp.model
            torch.save(hyp.model.state_dict(), 'best_hyp.model.pth')
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping. No improvement in validation loss.")
                break


print("\nTraining Time (in minutes) =", (time() - time0) / 60)

# Load the best hyp.model
hyp.model.load_state_dict(torch.load('best_hyp.model.pth'))

# Rest of your code for prediction and accuracy calculation...

# Predict on a single image from the validation set
images, labels = next(iter(valloader))
img = images[0].view(1, 784)

with torch.no_grad():
    logps = hyp.model(img.to(device))  # Move the image to the GPU if available

ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])  # Move probabilities back to the CPU
predicted_digit = probab.index(max(probab))
print("Predicted Digit =", predicted_digit)

correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        with torch.no_grad():
            logps = hyp.model(img.to(device))  # Move the image to the GPU if available

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])  # Move probabilities back to the CPU
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nhyp.model Accuracy =", (correct_count / all_count))

# Save the hyp.model to a file (note that you should specify a valid path)
torch.save(hyp.model.state_dict(), 'path_to_save_hyp.model.pth')

class MNISTModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # model layers using input_size, hidden_sizes  


def train_mnist(config):  
    model = MNISTModel(config["input_size"], config["hidden_sizes"], 10) 
    
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    
    for epoch in range(10):
        train(model, optimizer)
        val_acc = test(model)
        tune.report(mean_accuracy=val_acc)

        config = {
  "input_size": 784,
  "hidden_sizes": tune.choice([[64, 128], [512, 256]]), 
  "lr": tune.loguniform(1e-4, 1e-1)
}
        
        config = {
  "input_size": 784,
  "hidden_sizes": tune.choice([[64, 128], [512, 256]]), 
  "lr": tune.loguniform(1e-4, 1e-1)
}
        
        analysis = tune.run(train_mnist, config=config)
print(analysis.best_config)