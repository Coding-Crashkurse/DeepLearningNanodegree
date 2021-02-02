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
from torch import optim


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
optimizer = optim.SGD(model.parameters(), lr=0.04)

epochs = 40
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f'Training loss: {running_loss/len(trainloader)}')

torch.save(model.state_dict(), "C:/Users/User/Desktop/Canvas/model.pth")

model.load_state_dict(torch.load("C:/Users/User/Desktop/Canvas/model.pth"))
model.eval()


# Create prediction
images, labels = next(iter(trainloader))
images[0].squeeze().shape

images[0].view(784).shape

T = images[0]
T = T.unsqueeze(0)
T = torch.nn.functional.interpolate(T,size=(14,14), mode='bilinear')
T = T.squeeze(0)
T.size()

plt.imshow(images[0].numpy().squeeze(), cmap="Greys_r")

F.interpolate(images[0], (1, 14, 14))

img = images[0].view(1, 784)
img

with torch.no_grad():
    logits = model.forward(img)
    
ps = F.softmax(logits, dim=1)

import seaborn as sns
import numpy as np
import pandas as pd


ps_num = ps.numpy().flatten()
result = pd.concat([pd.Series(ps_num), pd.Series(np.arange(0, 10))], axis=1).rename(columns={0: "prediction", 1: "values"})

sns.barplot(data=result, y="prediction", x="values")       
        

def convert_scale(arr):
    result = (-1 + 2 * ((arr) - np.min(arr)) / (np.max(arr) - np.min(arr)))
    return(result)
    
convert_scale(np.array([0, 0, 0, 255, 255, 255]))

output = np.array([0, 0, 0, 255, 255, 255])
result = (-1 + 2 * ((output) - 0) / (255 - 0)) * -1;
result

desired = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])


x = torch.from_numpy(desired)
x.reshape((1, 3, 3)).shape
        

np.array([1,2,3,4,5]).argmax()
np.array([1,2,3,4,5]).max()

    