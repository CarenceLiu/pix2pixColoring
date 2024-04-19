'''
Wenrui Liu
2024-4-16

dataset to load CIFAR
'''
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class CIFARDataset(Dataset):
    def __init__(self, root_dir, split="train", transform_color=None, transform_black=None):
        self.root_dir = root_dir
        self.color_image_dir = os.path.join(self.root_dir, "images_%s"%(split))
        self.black_image_dir = os.path.join(self.root_dir, "images_%s_black"%(split))
        self.transform_black = transform_black
        self.transform_color = transform_color
        self.image_filename = [file for file in os.listdir(self.color_image_dir) if file.endswith('.png')][:4096]
        self.images_color = [Image.open(os.path.join(self.color_image_dir, file)).convert('RGB') for file in tqdm(self.image_filename, desc='read color image')]
        self.images_color = [self.transform_color(image) if self.transform_black else image for image in tqdm(self.images_color, desc="color image transform")]
        self.images_black = [Image.open(os.path.join(self.black_image_dir, file)) for file in tqdm(self.image_filename, desc='read black image')]
        self.images_black = [self.transform_black(image) if self.transform_black else image for image in tqdm(self.images_black, desc="black image transform")]

    def __len__(self):
        return len(self.images_color)

    def __getitem__(self, idx):
        return self.images_color[idx], self.images_black[idx]

if __name__ == "__main__":
    transformer_black = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    transformer_color = transforms.Compose([
        transforms.ToTensor(),
    ])
    full_train_data = CIFARDataset("../data", "train", transformer_color, transformer_black)
    train_size = int(0.9 * len(full_train_data))
    validation_size = len(full_train_data) - train_size
    train_data, validation_data = random_split(full_train_data, [train_size, validation_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle = True)
    validation_loader = DataLoader(validation_data, batch_size=256, shuffle=True)
    for batch_idx, (color_images, black_images) in enumerate(train_loader):
        print(batch_idx, color_images.size(), black_images.size())
        break