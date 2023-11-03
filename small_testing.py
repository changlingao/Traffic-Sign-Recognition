
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import Net
import torchvision.transforms as transforms

# params
number_of_classes = 43
classes = [str(i) for i in range(number_of_classes)]  # Generates a list from '0' to '42'
net_path = 'net.pth'
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
batch_size = 50

# load test images to device
test_path = './Small_Testing/'
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
