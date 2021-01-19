import torch

def activation(x):
    return 1/(1+torch.exp(-x))


torch.manual_seed(7)
features = torch.randn((1,5))

features

weights = torch.randn_like(features)
bias = torch.randn((1,1))


activation(torch.sum(features * weights) + bias)
activation(torch.mm(features, weights.view(5, 1)) + bias)

### Small multilayer network

torch.manual_seed(7)

features = torch.randn((1,3))

n_input = features.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))


h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
output

### Neural networks with PyTorch

from torchvision import datasets, transforms
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

trainset = datasets.MNIST("MNIST_data/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()


images[1].numpy().squeeze().shape
images[1].numpy().shape

plt.imshow(images[1].numpy().squeeze(), cmap="Greys_r")

### Solution

def activation(x):
    return 1/(1+torch.exp(-x))

inputs = images.view(images.shape[0], -1)


w1 = torch.randn(784, 256)
b1 = torch.randn(257)

w2 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)
out= torch.mm(h, w2) + b2
out[0:10]

## Softmax fehlt noch
from torch import nn
from torch import optim
import torch.nn.functional as F

model = nn.Sequential(nn.Linear(784, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 10),
              nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10
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

        
images, labels = next(iter(trainloader))

labels

img = images[0].view(1, 784)                
plt.imshow(images[0].numpy().squeeze(), cmap="Greys_r")

with torch.no_grad():
    logits = model.forward(img)      
    
ps = F.softmax(logits, dim=1)

        
        
        
        
        
        
        