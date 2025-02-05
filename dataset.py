# dataset.py
import os
import torch
from torch.utils.data import Dataset as TorchDataset

class CrackDataset(TorchDataset):
    """
    Custom Dataset for Crack Detection.
    
    The dataset loads positive and negative sample tensor files.
    """
    def __init__(self, root_dir="/home/wsuser/work", transform=None, train=True):
        positive_dir = os.path.join(root_dir, "Positive_tensors")
        negative_dir = os.path.join(root_dir, "Negative_tensors")

        positive_files = sorted([os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith(".pt")])
        negative_files = sorted([os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith(".pt")])
        
        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        # Interleave positive and negative files
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files 
        
        # Create labels: positive=1 for even-indexed files, negative=0 for odd-indexed files
        self.Y = torch.zeros(number_of_samples, dtype=torch.long)
        self.Y[::2] = 1  # positive images
        self.Y[1::2] = 0 # negative images

        if train:
            self.all_files = self.all_files[0:30000]
            self.Y = self.Y[0:30000]
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]
        
        self.transform = transform
        self.len = len(self.all_files)
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = torch.load(self.all_files[idx])
        y = self.Y[idx]
        if self.transform:
            image = self.transform(image)
        return image, y

if __name__ == "__main__":
    # Quick test to ensure the dataset loads correctly
    dataset = CrackDataset(train=True)
    print("Number of training samples:", len(dataset))
