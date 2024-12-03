import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict
import os

class CustomFacialDataset:
    def __init__(self, csv_file, FAC=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and emotions
            FAC (boolean): State whether the dataset is prepared for the FAC classificatio
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.emotions_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.FAC = FAC
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Handle error as loading image
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            img_path = self.data.iloc[idx, 1]
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Get emotion label
            labels = self.data['high-level-emotion']
            
            # If FAC => get all FAC labels
            if self.FAC:
                labels =  torch.FloatTensor(self.data.iloc[idx, 3:].values)
                    
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return the data point as tensor of scalar zero
            return {
                'image': torch.zeros((3, 224, 224)),
                'labels': torch.zeros(len(self.data.iloc[0, 3:])) if self.FAC else torch.tensor(0) # Handle singular label
            }
            

class CustomLoader:
    def __init__(self, dataset: CustomFacialDataset, batch_size: int = 32, shuffle: bool = True):
        """
        Args:
            dataset (CustomFacialDataset): Dataset of images and labels
            batch_size (int): State the size of data batches
            shuffle (boolean): Optional shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))

    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.current_idx >= len(self.data):
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data = [self.dataset[idx] for idx in batch_indices]
        
        self.current_idx += self.batch_size
        
        return {
            'images': torch.stack([item['image'] for item in batch_data]),
            'labels': torch.stack([item['labels'] for item in batch_data])
        }
