import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layerN1 = 512
layerN2 = 256
layerN3 = 128
layerN4 = 64

neuron = input_size = 784
hidden_sizes = [layerN1, layerN2, layerN3, layerN4]
output_size = 10
NL = 0
count = 0

def increment():
   global NL, count
   if count % 2 == 0:
      NL += 1
   count += 1


linear = nn.Linear(hidden_sizes[NL], hidden_sizes[NL]).to(device)

relu= nn.ReLU()

count_layer = 1

linear_product = count_layer * linear 
relu_product = count_layer * relu

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]).to(device),
    nn.ReLU(),


    linear_product,
    relu_product,

    nn.Linear(hidden_sizes[1], hidden_sizes[2]).to(device),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]).to(device),
    nn.ReLU(),


    nn.Linear(hidden_sizes[3], output_size).to(device),
    nn.LogSoftmax(dim=1)
).to(device)
