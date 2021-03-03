import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index])
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.files)
