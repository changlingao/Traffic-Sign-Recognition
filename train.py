# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# When import a Python module, any executable code that's not contained within functions or classes will be executed.

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from net import Net
import torchvision.transforms as transforms


number_of_classes = 43
classes = [str(i) for i in range(number_of_classes)]  # Generates a list from '0' to '42'

net_path = 'net.pth'


# tuning: data augmentation technique (random flip...)
transform = transforms.Compose([
    # resize and normalize folloe the value in evaluation/test.py 
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.RandomPerspective(0.3, 0.1) # data augmentation
])

# tuning: 16, 32, 64 ...
batch_size = 32
# 1000 training examples, batch size is 50, take 20 iterations to complete 1 epoch.


# load train data
train_path = './GTSRB_Final_Training_Images/Final_Training/Images/'
trainset = ImageFolder(root=train_path, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# construct net
net = Net(number_of_classes)

criterion = nn.CrossEntropyLoss()
# tuning learning rate: 0.001
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# actual training
for epoch in range(3):  # tuning the loop over times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        # move to GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print every epoch
        n = len(trainset) // batch_size
        if i % n == (n - 1):
            print(f'[{epoch + 1}, {i + 1 :5d}] loss: {running_loss / n :.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), net_path)

