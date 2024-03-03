import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time
from collections import OrderedDict
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# layer = 0
# count_layer = layer[:]

count_layer = 10

# define hidden_layers 
# automatische Variante
hidden_sizes=[]
for i in range(1, count_layer+1):
  hidden_sizes.append(64*i)
hidden_sizes.reverse()
print("Running with count_layers= " + str(count_layer) + " and hidden_sizes= " + str(hidden_sizes)+"\n") 

# Oder: Variante zum manuell nachsteuern
# if count_layer ==1:
#  hidden_sizes = [64]
# elif count_layer ==2:
#  hidden_sizes = [128, 64]
# elif count_layer ==3:
#  hidden_sizes = [256, 128, 64]
# elif count_layer ==4:
#  hidden_sizes = [512, 256, 128, 64]
# elif count_layer ==5:
#  hidden_sizes = [1024, 512, 256, 128, 64]

neuron = input_size = 784
output_size = 10
NL = 0
count = 0

# def increment():
#   global NL, count
#   if count % 2 == 0:
#      NL += 1
#   count += 1



# linear = nn.Linear(hidden_sizes[NL], hidden_sizes[NL]).to(device)
# relu= nn.ReLU()



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