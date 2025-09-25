"""
Main Training Script for CNN-based Pneumonia Detection
Orchestrates the complete training pipeline with comprehensive evaluation
"""

import torch
import os
import time
import json
import numpy as np

from dataset import create_data_loaders
from cnn_model import create_model, print_model_summary
from train import (
    setup_training_components, train_one_epoch, validate,
    save_checkpoint, get_current_lr, calculate_binary_metrics
)
from utils import plot_training_curves, save_training_history, setup_environment

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cnn_config import *

def print_training_summary(model, class_counts):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("🚀 CNN PNEUMONIA DETECTION TRAINING")
    print("="*60)
    
    print(f"📊 Dataset Information:")
    normal_count, pneumonia_count = class_counts
    print(f"   • Normal cases: {normal_count:,}")
    print(f"   • Pneumonia cases: {pneumonia_count:,}")
    print(f"   • Total samples: {normal_count + pneumonia_count:,}")
    print(f"   • Class ratio: {normal_count/pneumonia_count:.1f}:1")
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   • Architecture: {model.architecture.value}")
    print(f"   • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Learning rate: {LEARNING_RATE}")
    print(f"   • Weight decay: {WEIGHT_DECAY}")
    print(f"   • Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   • Results directory: {RESULTS_DIR}")
    
    print("="*60)

def train_cnn(architecture=DEFAULT_ARCHITECTURE, augmentation_strategy='none'):
    """
    Train the CNN model and save training results
    
    Args:
        architecture: CNNArchitecture enum value
        augmentation_strategy: 'none', 'light', or 'aggressive'
        
    Returns:
        tuple: (best_model_path, train_metrics, val_metrics, class_counts)
    """
    print("🏥 Starting CNN Pneumonia Detection Training...")
    
    # Setup environment
    device = setup_environment(results_dir=RESULTS_DIR, checkpoint_dir=CHECKPOINT_DIR)
    
    # Create data loaders
    print("\n📁 Loading dataset...")
    train_loader, val_loader, test_loader, class_counts = create_data_loaders(
        augmentation_strategy=augmentation_strategy,
        batch_size=BATCH_SIZE
    )
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_model(device=device, architecture=architecture)
    print_model_summary(model)
    
    # Print training summary
    print_training_summary(model, class_counts)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(
        model, 
        negative_class_count=class_counts[0],
        positive_class_count=class_counts[1],
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Training tracking
    train_metrics = {
        'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [], 'learning_rate': []
    }
    val_metrics = {
        'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [], 'auc': [], 
        'sensitivity': [], 'specificity': [], 'samples': [], 'time': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    print(f"\n🏃 Starting training for {EPOCHS} epochs...")
    training_start_time = time.time()

    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Training
        train_epoch_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validation
        val_epoch_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_epoch_metrics['loss'])
        current_lr = get_current_lr(optimizer)
        
        # Store metrics
        for key in train_metrics:
            if key == 'learning_rate':
                train_metrics[key].append(current_lr)
            elif key in ['precision', 'recall']:
                train_metrics[key].append(train_epoch_metrics[key])
            else:
                train_metrics[key].append(train_epoch_metrics[key])
        
        for key in val_metrics:
            if key in ['samples', 'time']:
                val_metrics[key].append(val_epoch_metrics.get(key, 0))
            else:
                val_metrics[key].append(val_epoch_metrics[key])
        
        # Check for best model with min_delta
        improvement = val_epoch_metrics['f1_score'] - best_f1
        if improvement > MIN_DELTA:
            best_f1 = val_epoch_metrics['f1_score']
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = os.path.join(RESULTS_DIR, 'best_model.pt')
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_epoch_metrics, val_epoch_metrics,
                best_model_path, is_best=True
            )
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 Training completed!")
    print(f"   • Total time: {total_training_time/60:.1f} minutes")
    print(f"   • Best F1: {best_f1:.4f} (Epoch {best_epoch + 1})")
    
    # Save training curves and history
    curves_path = os.path.join(RESULTS_DIR, 'learning_curves.png')
    plot_training_curves(train_metrics, val_metrics, save_path=curves_path)
    
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    save_training_history(train_metrics, val_metrics, history_path, best_epoch, best_f1, total_training_time)
    
    # Save final training metrics
    final_metrics = {
        'best_epoch': best_epoch + 1,
        'best_f1_score': best_f1,
        'final_train_loss': train_metrics['loss'][-1],
        'final_val_loss': val_metrics['loss'][-1],
        'total_training_time_minutes': total_training_time / 60,
        'total_epochs': len(train_metrics['loss'])
    }
    
    metrics_path = os.path.join(RESULTS_DIR, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"📝 Training metrics saved to: {metrics_path}")

    # Load the best model before returning
    best_model_path = os.path.join(RESULTS_DIR, 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Updated to best model before return. Best model path:  {best_model_path}")
    
    return model, train_metrics, val_metrics, class_counts, test_loader
