import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time
from collections import OrderedDict
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

count_layer = 3
layer = 0
count_layer = layer[:]


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



# prepare args for sequental
od = OrderedDict()

od['convfirst'] = nn.Linear(input_size, hidden_sizes[0]).to(device)
od['relufirst'] = nn.ReLU()

for x in range(1, count_layer):
    od['conv' + str(x)] = nn.Linear(hidden_sizes[x-1], hidden_sizes[x]).to(device)
    od['relu' + str(x)] = nn.ReLU()

od['convlast'] = nn.Linear(hidden_sizes[x], output_size).to(device)
od['relulast'] = nn.LogSoftmax(dim=1)

model = nn.Sequential(od).to(device)

