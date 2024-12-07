import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit

def split_dataset(csv_path, train_size=0.8, random_state=42):
    data = pd.read_csv(csv_path)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data, data['high_level_emotion']))
    return data.iloc[train_idx], data.iloc[test_idx]
class EmotionFACsDataset:
    def __init__(self, data_df, transform=None, emotion_map=None):
        self.data = data_df
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Map emotions to integers
        self.emotion_map = emotion_map or {
            'negative': 0, 'positive': 1, 'surprise': 2
        }
        
        # FACs columns are named 'AU1', 'AU2', etc.
        self.fac_columns = [col for col in self.data.columns if col.startswith('AU')]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and transform image
        image = Image.open(f'data/{row['filepath']}').convert('RGB')
        image = self.transform(image)
        
        # Get emotion label
        emotion = torch.tensor(self.emotion_map[row['high_level_emotion']], dtype=torch.long)
        
        # Get FACs labels ( binary values)
        facs = torch.tensor((row[self.fac_columns].values.astype(int)), dtype=torch.float32)
        
        return image, emotion, facs

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            batch_images = []
            batch_emotions = []
            batch_facs = []
            
            for idx in batch_indices:
                image, emotion, facs = self.dataset[idx]
                batch_images.append(image)
                batch_emotions.append(emotion)
                batch_facs.append(facs)
            
            yield (torch.stack(batch_images), 
                  torch.stack(batch_emotions),
                  torch.stack(batch_facs))
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size