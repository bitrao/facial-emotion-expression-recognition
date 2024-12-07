import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime
import os


class EmotionFACsTrainer:
    def __init__(
        self, 
        model, 
        model_name,
        train_loader, 
        val_loader,
        device='cuda',
        lr=1e-4,
        emotion_loss_weight=0.5,
        output_dir = None,
        num_epochs=50,
        patience = None,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.emotion_loss_weight = emotion_loss_weight
        self.num_epochs = num_epochs
        self.patience = patience
        self.learning_rate= lr
        
        # Training history
        self.history = {
            'train_emotion_loss': [], 'train_facs_loss': [],
            'val_emotion_loss': [], 'val_facs_loss': [],
            'train_emotion_f1': [], 'train_facs_f1': [],
            'val_emotion_f1': [], 'val_facs_f1': []
        }

        
        self.output_dir =os.path.join(output_dir or 'models/outputs/' , datetime.now().strftime('%Y%m%d'))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize training components
        self.setup_training()
    def setup_training(self):
        """Setup training components."""
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.facs_criterion = nn.BCEWithLogitsLoss()
        
        # Different learning rates for different layers
        # Apply AdamW for better generalization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Adjust learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )
    
    def train_epoch(self):
        self.model.train()
        total_emotion_loss = 0
        total_facs_loss = 0
        all_emotion_preds = []
        all_emotion_labels = []
        all_facs_preds = []
        all_facs_labels = []
        
        for images, emotions, facs in tqdm(self.train_loader):
            images = images.to(self.device)
            emotions = emotions.to(self.device)
            facs = facs.to(self.device)
            
            self.optimizer.zero_grad()
            
            emotion_preds, facs_preds = self.model(images)
            
            emotion_loss = self.emotion_criterion(emotion_preds, emotions)
            facs_loss = self.facs_criterion(facs_preds, facs)
            
            # Combined loss
                   
            loss = emotion_loss + facs_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_emotion_loss += emotion_loss.item()
            total_facs_loss += facs_loss.item()
            
            # Store predictions for metrics
            all_emotion_preds.extend(emotion_preds.argmax(1).cpu().numpy())
            all_emotion_labels.extend(emotions.cpu().numpy())
            all_facs_preds.extend((facs_preds > 0).float().cpu().numpy())
            all_facs_labels.extend(facs.cpu().numpy())
        
        # Calculate metrics
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds,average='macro')
        facs_f1 = f1_score(np.array(all_facs_labels), np.array(all_facs_preds), average='macro')
        
        return {
            'emotion_loss': total_emotion_loss / len(self.train_loader),
            'facs_loss': total_facs_loss / len(self.train_loader),
            'emotion_f1': emotion_f1,
            'facs_f1': facs_f1,
            'total_loss': loss
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_emotion_loss = 0
        total_facs_loss = 0
        all_emotion_preds = []
        all_emotion_labels = []
        all_facs_preds = []
        all_facs_labels = []
        
        for images, emotions, facs in self.val_loader:
            images = images.to(self.device)
            emotions = emotions.to(self.device)
            facs = facs.to(self.device)
            
            emotion_preds, facs_preds = self.model(images)
            
            emotion_loss = self.emotion_criterion(emotion_preds, emotions)
            facs_loss = self.facs_criterion(facs_preds, facs)
            loss = emotion_loss + facs_loss
            
            total_emotion_loss += emotion_loss.item()
            total_facs_loss += facs_loss.item()
            
            all_emotion_preds.extend(emotion_preds.argmax(1).cpu().numpy())
            all_emotion_labels.extend(emotions.cpu().numpy())
            all_facs_preds.extend((facs_preds > 0).float().cpu().numpy())
            all_facs_labels.extend(facs.cpu().numpy())
        
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
        facs_f1 = f1_score(np.array(all_facs_labels), np.array(all_facs_preds), average='macro')
        
        return {
            'emotion_loss': total_emotion_loss / len(self.val_loader),
            'facs_loss': total_facs_loss / len(self.val_loader),
            'emotion_f1': emotion_f1,
            'facs_f1': facs_f1,
            'total_loss': loss
        }
    
    def train(self, num_epochs):
        
        best_loss = float('inf')
        best_val_metrics = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            val_loss = val_metrics['total_loss']
            
            # Store metrics
            self.history['train_emotion_loss'].append(train_metrics['emotion_loss'])
            self.history['train_facs_loss'].append(train_metrics['facs_loss'])
            self.history['train_emotion_f1'].append(train_metrics['emotion_f1'])
            self.history['train_facs_f1'].append(train_metrics['facs_f1'])
            
            self.history['val_emotion_loss'].append(val_metrics['emotion_loss'])
            self.history['val_facs_loss'].append(val_metrics['facs_loss'])
            self.history['val_emotion_f1'].append(val_metrics['emotion_f1'])
            self.history['val_facs_f1'].append(val_metrics['facs_f1'])
            
            print(f"Train - Emotion Loss: {train_metrics['emotion_loss']:.4f}, "
                  f"FACS Loss: {train_metrics['facs_loss']:.4f}")
            print(f"Train - Emotion F1: {train_metrics['emotion_f1']:.4f}, "
                  f"FACS F1: {train_metrics['facs_f1']:.4f}")
            print(f"Val - Emotion Loss: {val_metrics['emotion_loss']:.4f}, "
                  f"FACS Loss: {val_metrics['facs_loss']:.4f}")
            print(f"Val - Emotion F1: {val_metrics['emotion_f1']:.4f}, "
                  f"FACS F1: {val_metrics['facs_f1']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model 
            if val_loss < best_loss:
                best_loss = val_loss
                best_val_metrics = val_metrics
                self.save_checkpoint(epoch)
                counter = 0
            else:
                counter += 1
            
            # Early stopping
            if (self.patience is not None) and (counter >= self.patience):
                print(f"Early stop at epoch {epoch}")
                print(f"Best Val - Emotion Loss: {best_val_metrics['emotion_loss']:.4f}, "
                    f"Best FACS Loss: {best_val_metrics['facs_loss']:.4f}")
                print(f"Best Val - Emotion F1: {best_val_metrics['emotion_f1']:.4f}, "
                    f"Best FACS F1: {best_val_metrics['facs_f1']:.4f}")
                break
            

    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'{self.model_name or self.model.backbone_name}.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'backbone_name': self.model.backbone_name,
            'num_emotions': self.model.num_emotions,
            'num_facs': self.model.num_facs,
        }
        torch.save(checkpoint, checkpoint_path)

    def plot_metrics(self):
        epochs = range(1, len(self.history['train_emotion_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(epochs, self.history['train_emotion_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.history['val_emotion_loss'], 'r-', label='Val')
        ax1.set_title('Emotion Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        ax2.plot(epochs, self.history['train_facs_loss'], 'b-', label='Train')
        ax2.plot(epochs, self.history['val_facs_loss'], 'r-', label='Val')
        ax2.set_title('FACS Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        # Plot metrics
        ax3.plot(epochs, self.history['train_emotion_f1'], 'b-', label='Train')
        ax3.plot(epochs, self.history['val_emotion_f1'], 'r-', label='Val')
        ax3.set_title('Emotion F1')
        ax3.set_xlabel('Epoch')
        ax3.legend()
        
        ax4.plot(epochs, self.history['train_facs_f1'], 'b-', label='Train')
        ax4.plot(epochs, self.history['val_facs_f1'], 'r-', label='Val')
        ax4.set_title('FACS F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()