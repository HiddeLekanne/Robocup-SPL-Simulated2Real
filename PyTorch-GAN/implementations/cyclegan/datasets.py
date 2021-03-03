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

        # print(os.path.join(root, mode) + "/*.*")
        if mode == "train":
            self.files_A = sorted(glob.glob(os.path.join(root, mode + "A") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, mode + "B") + "/*.*"))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, "testA") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "testB") + "/*.*"))
    
    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        # print(len(self.files_B))
        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else: 
            img_B = Image.open(self.files_B[index % len(self.files_B)])

        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        # print(self.transform)
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
