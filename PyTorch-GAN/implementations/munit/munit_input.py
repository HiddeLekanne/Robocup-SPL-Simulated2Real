import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_input import *

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
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=5, help="dimensionality of the style code")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("%s" % opt.output_location, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

criterion_recon = torch.nn.L1Loss()

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)

if cuda:
    Enc1 = Enc1.cuda()
    Dec2 = Dec2.cuda()

Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, opt.model_number)))
Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, opt.model_number)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(opt.input_location, transforms_=transforms_),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

print(len(dataloader.dataset))
data = iter(dataloader)
for i in range(len(dataloader.dataset)):
    """Saves a generated sample from the validation set"""
    img = next(data)
    # print(img.unsqueeze(0).size())
    X1 = img.repeat(opt.style_dim, 1, 1, 1)
    X1 = Variable(X1.type(Tensor))
    # Get random style codes
    s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
    s_code = Variable(Tensor(s_code))
    # Generate samples
    c_code_1, _ = Enc1(X1)
    X12 = Dec2(c_code_1, s_code)
    # Concatenate samples horisontally
    name = dataloader.dataset.files[i].split("/")[-1]

    for i, sample in enumerate(X12):
        tmp_name =  name.split(".")[0] + "_" + str(i) + "." + name.split(".")[-1]

        sample = F.interpolate(sample.unsqueeze(0), size=(480,640), mode='bicubic')
        # sample = transform.resize(sample.unsqueeze(0), (480, 640))
        save_image(sample, opt.output_location + "/" + tmp_name, normalize=True)
