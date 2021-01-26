# Neural networks with PyTorch

import torch

def activation(x):
    return 1/(1+torch.exp(-x))


torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)


bias = torch.randn((1,1))
bias

features
weights.reshape(5, 1)

y = activation(torch.mm(features, weights.view(5, 1)) + bias)
y

# Multilay NN

features = torch.randn((1,3))

n_input = features.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

activation(torch.mm(features, W1) + B1)

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
output

# mnist
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

def activation(x):
    return 1/(1+torch.exp(-x))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST("MNIST_data/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()    

plt.imshow(images[8].numpy().squeeze(), cmap="Greys_r")


inputs = images.view(images.shape[0], -1)
inputs

w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

probabilities.sum(dim=1)

### mit nn.Module
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
        
model = Network()
        
## Functional

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x
    
        
## Mit backward propagation
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST("MNIST_data/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))    
        
criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)

print(loss)

model[0].weight.grad # None
loss.backward()
model[0].weight.grad      
        

from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    