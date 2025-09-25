"""
Main Training Script for ViT-based Pneumonia Detection
Orchestrates the complete training pipeline with freeze-then-unfreeze strategy
"""

import torch
import torch.nn as nn
import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from dataset import create_data_loaders
from model import create_model
from train import (
    train_one_epoch, validate, setup_training_components, 
    save_checkpoint, get_current_lr
)
from utils import plot_training_curves, save_training_history, setup_environment

import sys
import os
import torch.optim as optim

# Add parent directory to path for new config structure  
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vit_config import *

def update_optimizer_for_unfreeze(model, optimizer, unfreeze_lr_factor=0.1):
    """
    Unique for ViT model
    Update optimizer when transitioning from frozen to unfrozen
    
    Args:
        model: The ViT model
        optimizer: Current optimizer
        unfreeze_lr_factor: Factor to reduce learning rate by
        
    Returns:
        torch.optim.Optimizer: Updated optimizer
    """
    if not model.is_frozen():
        # Create new optimizer for unfrozen model with reduced learning rate (best  practice for fine tuning)
        new_lr = LEARNING_RATE * unfreeze_lr_factor
        
        new_optimizer = optim.AdamW(
            model.parameters(),
            lr=new_lr,
            weight_decay=WEIGHT_DECAY
        )
        
        print(f"🔄 Updated optimizer for unfrozen training:")
        
        return new_optimizer
    
    return optimizer

def print_training_summary(strategy_info, total_params, trainable_params):
    """Print comprehensive training summary"""
    print("\n" + "="*80)
    print("🚀 TRAINING SUMMARY")
    print("="*80)
    print(f"📊 Model Configuration:")
    print(f"   Architecture: {MODEL_NAME}")
    print(f"   Input: 1-channel grayscale ({IMAGE_SIZE}x{IMAGE_SIZE})")
    print(f"   Classes: {NUM_CLASSES} (NORMAL, PNEUMONIA)")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print(f"\n🎯 Training Strategy:")
    print(f"   Strategy: {strategy_info['strategy']}")
    print(f"   Total epochs: {strategy_info['total_epochs']}")
    print(f"   Frozen phase: {strategy_info['freeze_epochs']} epochs")
    print(f"   Unfrozen phase: {strategy_info['unfreeze_epochs']} epochs")
    print(f"   Transition at epoch: {strategy_info['unfreeze_at_epoch']}")
    
    print(f"\n⚙️ Hyperparameters:")
    print(f"   Learning rate: {LEARNING_RATE:.2e}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Weight decay: {WEIGHT_DECAY:.2e}")
    print(f"   Gradient clipping: {GRAD_CLIP_VALUE}")
    print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    print("="*80)

def train_vit(unfreeze_epochs = 1, augmentation_strategy='none'):
    """
    Train the ViT model with freeze-then-unfreeze strategy
    
    Returns:
        tuple: (model, train_history, val_history, test_loader, best_epoch, strategy_info)
    """
    print("2")
    print("🏥 Starting ViT Pneumonia Detection Training...")
    
    # 1. Setup Environment
    device = setup_environment(results_dir=RESULTS_DIR, checkpoint_dir=CHECKPOINT_DIR)
    
    # 2. Data Loading
    print("\n📊 Loading and preparing data...")
    train_loader, val_loader, test_loader, class_counts = create_data_loaders(
        augmentation_strategy=augmentation_strategy,
        batch_size=BATCH_SIZE
    )
    
    # 3. Model & Optimizer Setup
    print("\n🤖 Creating model...")
    
    # Freeze-then-unfreeze strategy configuration 
    FREEZE_EPOCHS = EPOCHS - unfreeze_epochs
    TOTAL_EPOCHS = EPOCHS  # Total training epochs
    
    print(TOTAL_EPOCHS)

    # Create model with initial freezing
    model = create_model(freeze_backbone=True, num_classes=1)
    model = model.to(device)
    
    # Setup freeze-then-unfreeze strategy
    strategy_info = model.setup_freeze_then_unfreeze(FREEZE_EPOCHS, TOTAL_EPOCHS)
    
    # Setup initial training components
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(
        model, 
        negative_class_count=class_counts[0],
        positive_class_count=class_counts[1],
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Print training summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_training_summary(strategy_info, total_params, trainable_params)
    
    # 4. Two-Phase Training Loop
    print(f"\n🏃 Starting two-phase training: {FREEZE_EPOCHS} frozen + {TOTAL_EPOCHS-FREEZE_EPOCHS} unfrozen epochs...")
    
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
    
    training_start_time = time.time()
    
    # PHASE 1: Frozen Backbone Training (with early stopping)
    print(f"\n🧊 PHASE 1: Frozen backbone training ({FREEZE_EPOCHS} epochs)")
    print("="*60)
    
    for epoch in range(FREEZE_EPOCHS):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{FREEZE_EPOCHS} (FROZEN) {'='*20}")
        
        # Training and validation
        train_epoch_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_epoch_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_epoch_metrics['loss'])
        current_lr = get_current_lr(optimizer)
        
        # Store metrics (same as before)
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
        
        # Early stopping check (only in frozen phase)
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping in frozen phase after {epochs_without_improvement} epochs without improvement")
            break
    
    # PHASE 2: Unfreeze and Full Fine-tuning (NO early stopping)
    print(f"\n🔥 PHASE 2: Unfrozen full model training ({TOTAL_EPOCHS-FREEZE_EPOCHS} epochs)")
    print("="*60)
    
    # Unfreeze the model
    model.unfreeze_backbone()
    
    # Update optimizer for unfrozen training
    optimizer = update_optimizer_for_unfreeze(model, optimizer, unfreeze_lr_factor=0.1)
    
    # Create new scheduler for updated optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA
    )
    
    # Reset early stopping for Phase 2 (but don't use it)
    phase2_start_epoch = len(train_metrics['loss'])
    
    for epoch in range(FREEZE_EPOCHS, TOTAL_EPOCHS):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{TOTAL_EPOCHS} (UNFROZEN) {'='*20}")
        
        # Training and validation
        train_epoch_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_epoch_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_epoch_metrics['loss'])
        current_lr = get_current_lr(optimizer)
        
        # Store metrics (same as before)
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
        
        # Check for best model (but NO early stopping in Phase 2)
        improvement = val_epoch_metrics['f1_score'] - best_f1
        if improvement > MIN_DELTA:
            best_f1 = val_epoch_metrics['f1_score']
            best_epoch = epoch
            
            # Save best model
            best_model_path = os.path.join(RESULTS_DIR, 'best_model.pt')
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_epoch_metrics, val_epoch_metrics,
                best_model_path, is_best=True
            )
    
    print(f"\n🎉 Two-phase training completed!")
    print(f"   • Phase 1 (Frozen): {min(len(train_metrics['loss']), FREEZE_EPOCHS)} epochs")
    print(f"   • Phase 2 (Unfrozen): {len(train_metrics['loss']) - min(len(train_metrics['loss']), FREEZE_EPOCHS)} epochs")
    
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
