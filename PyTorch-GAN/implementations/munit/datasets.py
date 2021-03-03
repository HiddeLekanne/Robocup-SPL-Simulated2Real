import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        # print(os.path.join(root, mode) + "/*.*")
        if mode == "train":
            self.files_A = sorted(glob.glob(os.path.join(root, mode + "A") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, mode + "B") + "/*.*"))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, "testA") + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "testB") + "/*.*"))
    
    def __getitem__(self, index):

        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_B)])

        # w, h = img.size
        # img_A = img.crop((0, 0, w / 2, h))
        # img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files_B)
