# python 3

''' script to load train and save models'''
import torch
import torch.optim as optim

from pt_toy_vision import Net


net = Net()
print(net)

# define parameters
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train model


# save entire model
# path to save model
PATH = '/home/taimur/Documents/Python Projects/pytorch_toy_model/saved_model/toy_vis.pt'

torch.save(net, PATH)