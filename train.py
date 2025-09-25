"""
Training Pipeline for CNN-based Pneumonia Detection
Implements training and validation logic with comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import os
import json
from datetime import datetime
from typing import Dict, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cnn_config import *

def setup_training_components(model, negative_class_count=DEFAULT_NORMAL_COUNT,
                            positive_class_count=DEFAULT_PNEUMONIA_COUNT,
                            learning_rate=LEARNING_RATE, device='cpu'):
    """
    Setup training components for binary pneumonia classification
    
    Args:
        model: PyTorch model
        negative_class_count: Number of normal (negative) samples in dataset
        positive_class_count: Number of pneumonia (positive) samples in dataset
        learning_rate: Initial learning rate for Adam optimizer
        device: Device to move pos_weight to
        
    Returns:
        Tuple of (loss_function, optimizer, scheduler)
    """
    print("🔧 Setting up training components...")
    

    pos_weight = torch.tensor(negative_class_count / positive_class_count, dtype=torch.float32)
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    print(f"   Class distribution:")
    print(f"   - Normal cases: {negative_class_count}")
    print(f"   - Pneumonia cases: {positive_class_count}")
    print(f"   - Calculated pos_weight: {pos_weight.item():.4f}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR
    )
    
    return criterion, optimizer, scheduler

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                   clip_grad_norm=GRAD_CLIP_VALUE, log_interval=LOG_INTERVAL):
    """
    Train the model for one epoch
    
    Args:
        model: The CNN model
        train_loader: DataLoader for training data
        optimizer: Optimizer (Adam)
        criterion: Loss function (with class weights)
        device: Device to run training on
        epoch: Current epoch number
        clip_grad_norm: Gradient clipping value
        log_interval: How often to log progress
        
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    
    # Initialize metrics tracking
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    start_time = time.time()
    
    print(f"🏃 Training Epoch {epoch + 1}")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        batch_size = images.size(0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Calculate metrics
        probabilities = torch.sigmoid(outputs).detach()
        predictions = (probabilities > 0.5).float()
        correct = (predictions == labels).sum().item()
        
        # Update running statistics
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct_predictions += correct
        
        # Store for comprehensive metrics
        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_probabilities.extend(probabilities.cpu().numpy().flatten())
        
    
    # Calculate epoch metrics
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    # Calculate comprehensive metrics
    metrics = calculate_binary_metrics(
        np.array(all_labels), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    metrics['loss'] = avg_loss
    metrics['epoch_time'] = epoch_time
    
    print(f"   ✅ Training completed | Time: {epoch_time:.1f}s | "
          f"Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | F1: {metrics['f1_score']:.4f}")
    
    return metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model
    
    Args:
        model: The CNN model  
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
        epoch: Current epoch number
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    start_time = time.time()
    
    print(f"🔍 Validating Epoch {epoch + 1}")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            batch_size = images.size(0)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Update running statistics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Store for comprehensive metrics
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probabilities.extend(probabilities.cpu().numpy().flatten())
    
    # Calculate epoch metrics
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    
    # Calculate comprehensive metrics
    metrics = calculate_binary_metrics(
        np.array(all_labels), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    metrics['loss'] = avg_loss
    metrics['epoch_time'] = epoch_time
    
    print(f"   ✅ Validation completed | Time: {epoch_time:.1f}s | "
          f"Loss: {avg_loss:.4f} | Acc: {metrics['accuracy']:.1f}% | "
          f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc']:.4f}")
    
    return metrics

def calculate_binary_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive binary classification metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Medical metrics (for pneumonia detection)
    sensitivity = recall  # True Positive Rate
    
    # Specificity = True Negative Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0  # If only one class present
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,  # Important for medical: ability to detect pneumonia
        'specificity': specificity,  # Important for medical: ability to correctly identify normal
        'auc': auc
    }

def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, 
                   checkpoint_path, is_best=False, improvement=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        improvement_str = f" (improvement: +{improvement:.4f})" if improvement else ""
        print(f"💾 New best model saved! Epoch {epoch + 1}, F1: {val_metrics['f1_score']:.4f}{improvement_str}")

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint

def get_current_lr(optimizer):
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]['lr']
