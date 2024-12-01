import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image
import logging
from datetime import datetime

class CNNTrainer:
    def __init__(self, 
                 train_dir,
                 val_dir,
                 num_epochs=10,
                 batch_size=32,
                 learning_rate=0.001,
                 image_size=224,
                 num_workers=2,
                 device=None,
                 model_name='resnet18',
                 output_dir='outputs'):
        """
        Initialize the CNN trainer.
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Initial learning rate
            image_size (int): Size to resize images to
            num_workers (int): Number of data loading workers
            device (str): Device to use ('cuda' or 'cpu')
            model_name (str): Name of the model to use
            output_dir (str): Directory to save outputs
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.num_workers = num_workers
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Create output directory
        self.output_dir = os.path.join(output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize training components
        self.setup_transforms()
        self.setup_datasets()
        self.setup_model()
        self.setup_training()
        
        self.logger.info(f"Initialized CNNTrainer with device: {self.device}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")

    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('CNNTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(os.path.join(self.output_dir, 'training.log'))
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_transforms(self):
        """Setup data transforms."""
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def setup_datasets(self):
        """Setup datasets and dataloaders."""
        # Create datasets
        self.train_dataset = ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_dataset = ImageFolder(self.val_dir, transform=self.val_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.num_classes = len(self.train_dataset.classes)
        self.class_to_idx = self.train_dataset.class_to_idx

    def setup_model(self):
        """Setup model architecture."""
        if self.model_name == 'resnet18':
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model = self.model.to(self.device)

    def setup_training(self):
        """Setup training components."""
        self.criterion = nn.CrossEntropyLoss()
        
        # Different learning rates for different layers
        self.optimizer = optim.Adam([
            {'params': self.model.fc.parameters()},
            {'params': [p for name, p in self.model.named_parameters() if 'fc' not in name],
             'lr': self.learning_rate/10}
        ], lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(self.train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return train_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies[self.train_dataset.classes[i]] = accuracy
        
        return (val_loss / len(self.val_loader), 
                100. * correct / total,
                class_accuracies)

    def train(self):
        """Complete training pipeline."""
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, class_accuracies = self.validate()
            
            # Log results
            self.logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            self.logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            self.logger.info('\nPer-class Validation Accuracy:')
            for class_name, accuracy in class_accuracies.items():
                self.logger.info(f'{class_name}: {accuracy:.2f}%')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)

    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'class_mapping': self.class_to_idx
        }, checkpoint_path)
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')

    def predict(self, image_path):
        """Predict class for a single image."""
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = outputs.max(1)
        
        # Map index to class name
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        predicted_class = idx_to_class[predicted.item()]
        
        return predicted_class

def main():
    # Example usage
    trainer = CNNTrainer(
        train_dir='path/to/train',
        val_dir='path/to/val',
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Train the model
    trainer.train()
    
    # Make a prediction
    prediction = trainer.predict('path/to/test/image.jpg')
    print(f'Predicted class: {prediction}')

if __name__ == '__main__':
    main()