import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image
import random

class AIArtBenchDataset(Dataset):
    def __init__(self, root, for_training=True, transforms=None, target_transforms=None):
        self.for_training = for_training
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        if self.for_training:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')
        
        self.fnames_list = []
        for directory in os.listdir(self.root):
            for image in os.listdir(os.path.join(self.root, directory)):
                self.fnames_list.append(os.path.join(directory, image))

        # random.shuffle(self.fnames_list)

    def __len__(self):
        return len(self.fnames_list)

    def __getitem__(self, idx):
        img_name = self.fnames_list[idx]
        img_path = os.path.join(self.root, img_name)
        image = read_image(img_path)

        if 'AI' in img_name:
            if 'SD' in img_name:
                label = 0 # 0 for standard diffusion art
            else:
                label = 1 # 1 for latent diffusion art
        else:
            label = 2 # 2 for real art
        
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        
        return image, label