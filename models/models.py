import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EmotionFACsNet(nn.Module):
    def __init__(self, num_emotions=7, num_facs=7, backbone_name="resnet50", pretrained=False):
        super(EmotionFACsNet, self).__init__()
        
        # Dictionary of backbone feature dimensions
        self.feature_dims = {
            'resnet18': 512,
            'resnet50': 2048,
            'vgg16': 512 * 7 * 7,
        }
        
        # Initialize backbone
        self.backbone = self._get_backbone(backbone_name, pretrained)
        feature_dim = self.feature_dims[backbone_name]
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions)
        )
        
        # FACS detection head
        self.facs_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_facs)
        )
    
    def _get_backbone(self, backbone_name, pretrained=False):
        """
        Args:
            backbone_name (str): Name of the backbone 
            pretrained (bool): Come with weight or not
        """
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            return nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            return nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
            return nn.Sequential(*list(backbone.features), nn.Flatten())
        
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Get predictions from both heads
        emotion_preds = self.emotion_head(features)
        facs_preds = self.facs_head(features)
        
        return emotion_preds, facs_preds


def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss functions
    emotion_criterion = nn.CrossEntropyLoss()
    facs_criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, emotion_labels, facs_labels) in enumerate(train_loader):
            images = images.to(device)
            emotion_labels = emotion_labels.to(device)
            facs_labels = facs_labels.to(device)
            
            # Forward pass
            emotion_preds, facs_preds = model(images)
            
            # Calculate losses
            emotion_loss = emotion_criterion(emotion_preds, emotion_labels)
            facs_loss = facs_criterion(facs_preds, facs_labels)
            
            # Combine losses with weights
            total_loss = emotion_loss + 0.5 * facs_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {total_loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_emotions = 0
        total = 0
        
        with torch.no_grad():
            for images, emotion_labels, facs_labels in val_loader:
                images = images.to(device)
                emotion_labels = emotion_labels.to(device)
                facs_labels = facs_labels.to(device)
                
                emotion_preds, facs_preds = model(images)
                
                # Calculate validation metrics
                emotion_loss = emotion_criterion(emotion_preds, emotion_labels)
                facs_loss = facs_criterion(facs_preds, facs_labels)
                val_loss += (emotion_loss + 0.5 * facs_loss).item()
                
                # Calculate emotion accuracy
                _, predicted = torch.max(emotion_preds.data, 1)
                total += emotion_labels.size(0)
                correct_emotions += (predicted == emotion_labels).sum().item()
        
        print(f'Epoch {epoch} completed')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Emotion Accuracy: {100 * correct_emotions / total:.2f}%')

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = EmotionFACSNet(num_emotions=7, num_aus=30)
    
    # Set up transforms for your data
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataloaders (you'll need to implement the actual data loading)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train the model
    # train_model(model, train_loader, val_loader)