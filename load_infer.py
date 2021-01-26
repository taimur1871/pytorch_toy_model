# python3 

''' script to load and run inference '''

import torch

# Specify a path
PATH = "/home/taimur/Documents/Python Projects/pytorch_toy_model/saved_model/toy_vis.pt"

# Load
model = torch.load(PATH)
model.eval()