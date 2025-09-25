"""
Utils file
"""

import torch
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from base_config import *


def plot_training_curves(train_history, val_history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, val_history['loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, train_history['accuracy'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, val_history['accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(epochs, train_history['f1_score'], 'b-', label='Training', linewidth=2)
    axes[0, 2].plot(epochs, val_history['f1_score'], 'r-', label='Validation', linewidth=2)
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # AUC-ROC (validation only)
    axes[1, 0].plot(epochs, val_history['auc'], 'r-', linewidth=2)
    axes[1, 0].set_title('AUC-ROC (Validation)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivity (validation only)
    axes[1, 1].plot(epochs, val_history['sensitivity'], 'r-', linewidth=2)
    axes[1, 1].set_title('Sensitivity - Pneumonia Detection (Validation)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sensitivity')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Specificity (validation only)
    axes[1, 2].plot(epochs, val_history['specificity'], 'r-', linewidth=2)
    axes[1, 2].set_title('Specificity - Normal Detection (Validation)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Specificity')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to {save_path}")
    plt.close()

def save_training_history(train_metrics, val_metrics, save_path, best_epoch, best_f1, total_training_time):
    """Save training history to JSON file"""
    history = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_epoch': best_epoch + 1,
        'best_val_f1_score': best_f1,
        'total_training_time_minutes': total_training_time / 60,
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'architecture': "",
            'timestamp': datetime.now().isoformat()
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"📝 Training history saved to: {save_path}")

def setup_environment(results_dir = "", checkpoint_dir = ""):

    """Setup training environment and device"""
    print("🔧 Setting up training environment...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
        
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    return device
 