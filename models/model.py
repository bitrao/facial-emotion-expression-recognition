import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

class EmotionFACsNet(nn.Module):
    def __init__(
        self, 
        num_emotions=3, 
        num_facs=15, 
        backbone_name="resnet50", 
        pretrained=False, 
        checkpoint=None, 
        emotion_labels=None
        ):
        super(EmotionFACsNet, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model from checkpoint
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location=self.device)
            num_emotions = checkpoint['num_emotions']
            num_facs = checkpoint['num_facs']
            backbone_name = checkpoint['backbone_name']
        
        # Dictionary of backbone feature dimensions
        self.feature_dims = {
            'resnet18': 512,
            'resnet50': 2048,
            'vgg16': 512 * 7 * 7,
        }
        
        self.emotion_labels = emotion_labels or {
             0 : 'negative',1: 'positive', 2: 'surprise'
        }
        
        # Transform to predict image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        self.num_emotions = num_emotions
        self.num_facs= num_facs
        
        # Initialize backbone
        self.backbone_name = backbone_name
        self.backbone = self._get_backbone(backbone_name, pretrained)
        feature_dim = self.feature_dims[backbone_name]
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions),
            nn.Softmax(dim=1)
        )
        
        # FACS detection head
        self.facs_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_facs)
        )
        
        # Load state dict if checkpoint was provided
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])
    
    def _get_backbone(self, backbone_name, pretrained=False):
        """
        Args:
            backbone_name (str): Name of the backbone 
            pretrained (bool): Come with weight or not
        """
        weights = "DEFAULT" if pretrained else None
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights = weights)
            return nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights = weights)
            return nn.Sequential(*list(backbone.children())[:-1])
            
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(weights = weights)
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


    def _get_binary_facs(self, facs_values):
        """
        Convert FACS values to binary based on threshold
        """
        return (facs_values > 0.5).astype(int)

    def predict_image(self, image_path):

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Move to device
        self.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # Set to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            emotion_probs, facs_values = self(image_tensor)
            
            # Get predicted emotion class
            emotion_class = torch.argmax(emotion_probs, dim=1).item()
            predicted_emotion = self.emotion_labels[emotion_class]
            
            # Convert to numpy
            facs_values = facs_values.cpu().numpy()[0]
            binary_facs = self._get_binary_facs(facs_values)
        
        return {
            'emotion': predicted_emotion,
            'facs_values': binary_facs
        }


