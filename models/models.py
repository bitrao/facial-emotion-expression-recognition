import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models


class EmotionRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognizer, self).__init__()
        # Load pre-trained ResNet
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    
class FACRecognizer(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)  