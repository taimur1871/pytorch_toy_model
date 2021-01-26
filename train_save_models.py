# python 3

''' script to load train and save models'''
import enum
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize

from pt_toy_vision import Net

torch.device('cpu')

net = Net()
print(net)

# define parameters
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# load image data
transform = transforms.Compose(
    [   transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(size=(32,32))])

dataset = torchvision.datasets.ImageFolder(
    '/home/taimur/Documents/Python Projects/pytorch_toy_model/pic_data',
    transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=2,
                                        shuffle=True)

classes = ('blade', 'cutter', 'other', 'top')


# train model
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get inputs
        inputs, labels = data

        # zero the gradients
        optimizer.zero_grad()

        # forward + backward+ loss
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('finished training')

# save entire model
# path to save model
PATH = '/home/taimur/Documents/Python Projects/pytorch_toy_model/saved_model/toy_vis.pt'

torch.save(net, PATH)