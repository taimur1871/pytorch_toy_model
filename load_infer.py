# python3 

''' script to load and run inference '''

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch.nn.functional as F


# Specify a path
PATH = "/home/taimur/Documents/Python Projects/pytorch_toy_model/saved_model/toy_vis.pt"
img_path = '/home/taimur/Pictures/Cutter Classification/train/chipped/11.jpg'

# load image
im = Image.open(img_path)
transform = Compose(
    [   ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Resize(size=(32,32))])

img = transform(im)

# Load model
model = torch.load(PATH)
model.eval()

#model
print(model(img))