import torchvision.transforms as transforms

# TODO: fine tuning
# crop size
# RandomHorizontalFlip: data augmentation technique
# Normalize value...
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# TODO: batch size
batch_size = 50



# TODO: change path
train_path = './GTSRB_Final_Training_Images/Final_Training/Images/'

# TODO: change constantly
number_of_classes = 43
classes = [str(i) for i in range(number_of_classes)]  # Generates a list from '0' to '42'

# TODO: change path
test_path = './Testing/'

net_path = 'net.pth'
