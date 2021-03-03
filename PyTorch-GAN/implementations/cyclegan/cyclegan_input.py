import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_input import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="simulated2real", help="name of the dataset")
parser.add_argument("--input_location", type=str, default="input_images/", help="input folder")
parser.add_argument("--output_location", type=str, default="output_images/", help="output folder")
parser.add_argument("--model_location", type=str, default="saved_models/", help="model location folder")
parser.add_argument("--model_number", type=int, default = 247, help="epoch number of the model")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=240, help="size of image height")
parser.add_argument("--img_width", type=int, default=320, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("%s" % opt.output_location, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()

# Load pretrained models
G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.model_number)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)) , Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(opt.input_location, transforms_=transforms_, unaligned=False),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

print(len(dataloader.dataset))

G_AB.eval()
data = iter(dataloader)
for i in range(len(dataloader.dataset)):
    img = next(data)
    real_A = Variable(img.type(Tensor))
    fake_B = G_AB(real_A)
    name = dataloader.dataset.files[i].split("/")[-1]

    fake_B = F.interpolate(fake_B, size=(480,640), mode='bicubic')
    save_image(fake_B, opt.output_location + "/" + name, normalize=True)