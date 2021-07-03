from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
import os
import torch

class LogoDataset(Dataset):
    def __init__(self, root, transform=None, download=False):
        
        original_test_dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
        original_train_dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform) 
        self.dataset = torch.utils.data.ConcatDataset([original_test_dataset, original_train_dataset])
        
        self.test_set_indices = np.arange(len(original_test_dataset))

        # these look useless, but are required by powerful-benchmarker
        self.labels = np.array([b for (a, b) in self.dataset.datasets[0].imgs + self.dataset.datasets[1].imgs])
        self.transform = transform 

    def get_split_indices(self, split_name):
        if split_name == "test":
            return self.test_set_indices
        return None
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
class LogoDatasetPlain(Dataset):
    def __init__(self, root, transform=None, download=False):
        
        self.dataset = datasets.ImageFolder(root, transform=transform)
        
        # these look useless, but are required by powerful-benchmarker
        self.labels = np.array([b for (a, b) in self.dataset.imgs])
        self.transform = transform 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]