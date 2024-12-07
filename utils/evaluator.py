import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score
from PIL import Image
from utils.data import EmotionFACsDataset, DataLoader

class EmotionFACsEvaluator:
    def __init__(self, model, val_csv_path, batch_size=32, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        val_df = pd.read_csv(val_csv_path)
        self.val_dataset = EmotionFACsDataset(val_df)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define FACS labels dictionary
        self.facs_labels = ['AU17', 'AU1', 'AU2', 'AU25', 'AU27', 'AU4', 'AU7', 'AU23', 'AU24',
       'AU6', 'AU12', 'AU15', 'AU14', 'AU11', 'AU26']

    def plot_confusion_matrix(self, confusion_mat, class_names):
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_mat, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Emotion Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()

    def generate_facs_summary(self, all_labels, all_preds):
        """Generate a detailed summary for each FACs unit."""
        summary = {}
        
        for i, au_name in enumerate(self.facs_labels):
            metrics = {}
            
            # Calculate metrics for this AU
            metrics['accuracy'] = accuracy_score(all_labels[:, i], all_preds[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[:, i],
                all_preds[:, i],
                average='binary'
            )
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                all_labels[:, i],
                all_preds[:, i]
            ).ravel()
            
            metrics['true_negative'] = tn
            metrics['false_positive'] = fp
            metrics['false_negative'] = fn
            metrics['true_positive'] = tp
            
            summary[au_name] = metrics
        
        return summary

    def generate_report(self):
        self.model.eval()
        
        all_preds = {'emotion': [], 'facs': []}
        all_labels = {'emotion': [], 'facs': []}
        
        with torch.no_grad():
            for images, emotion_labels, facs_labels in self.val_loader:
                images = images.to(self.device)
                
                emotion_probs, facs_values = self.model(images)
                emotion_preds = emotion_probs.argmax(1)
                facs_preds = (facs_values > 0.5).float()
                
                all_preds['emotion'].extend(emotion_preds.cpu().numpy())
                all_labels['emotion'].extend(emotion_labels.numpy())
                all_preds['facs'].extend(facs_preds.cpu().numpy())
                all_labels['facs'].extend(facs_labels.numpy())
            emotion_f1 = f1_score(all_labels['emotion'], all_preds['emotion'], average='macro')
            facs_f1 = f1_score(np.array(all_labels['facs']), np.array(all_preds['facs']), average='macro')
            print(f"Emotion F1: {emotion_f1:.4f}, "
                f"FACS F1: {facs_f1:.4f}")
        # Convert to numpy arrays
        for key in all_preds:
            all_preds[key] = np.array(all_preds[key])
            all_labels[key] = np.array(all_labels[key])
        
        report = {}
        
        # Emotion metrics
        emotion_names = list(self.model.emotion_labels.values())
        conf_matrix = confusion_matrix(
            all_labels['emotion'],
            all_preds['emotion']
        )
        
        report['emotion_confusion_matrix'] = self.plot_confusion_matrix(
            conf_matrix,
            emotion_names
        )
        
        report['emotion_classification_report'] = classification_report(
            all_labels['emotion'],
            all_preds['emotion'],
            target_names=emotion_names,
        )
        
        # FACS metrics
        facs_summary = self.generate_facs_summary(
            all_labels['facs'],
            all_preds['facs']
        )
        
        # Convert FACS summary to DataFrame
        summary_data = []
        for au_name, metrics in facs_summary.items():
            row = {
                'action_unit': au_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'true_positive': metrics['true_positive'],
                'false_positive': metrics['false_positive'],
                'false_negative': metrics['false_negative'],
                'true_negative': metrics['true_negative']
            }
            summary_data.append(row)
        
        report['facs_summary_df'] = pd.DataFrame(summary_data)
        
        return report

    def print_report(self, report):
        """Print a formatted version of the evaluation report."""
        print("=== Emotion Classification Results ===")
        print(report['emotion_classification_report'])
        
        print("\n=== FACs Detection Results ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        summary_df = report['facs_summary_df']
        print("\nFACs Performance Summary:")
        
        # Fixed column selection
        columns = ['action_unit', 'accuracy', 'precision', 'recall', 'f1']
        print(summary_df[columns].round(3))
        
