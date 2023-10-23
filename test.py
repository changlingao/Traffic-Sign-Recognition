
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import Net
from params import transform, batch_size, train_path, number_of_classes, classes, test_path, net_path


# load test images to device
testset = ImageFolder(root=test_path, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
dataiter = iter(testloader)
images, labels = next(dataiter) # no labels just for the next line to work somehow...
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
images = images.to(device)

# load net
net = Net(number_of_classes)
net.to(device)
net.load_state_dict(torch.load(net_path))

# predict
# print(images.is_cuda)
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
