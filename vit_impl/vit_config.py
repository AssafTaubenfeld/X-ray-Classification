"""
ViT-specific Configuration for Pneumonia Detection
Contains parameters specific to Vision Transformer implementation
"""
import os
from base_config import *

# =============================================================================
# VIT MODEL CONFIGURATION
# =============================================================================

# Pre-trained ViT model configuration
MODEL_NAME = "google/vit-base-patch16-224"
PATCH_SIZE = 16   # Patch size for ViT (16x16 patches)

# =============================================================================
# VIT TRAINING HYPERPARAMETERS
# =============================================================================

# Learning rate for fine-tuning (lower for pre-trained models)
LEARNING_RATE = 1e-5

# =============================================================================
# FREEZE/UNFREEZE STRATEGY
# =============================================================================

# Freeze strategy configuration
FREEZE_EPOCHS = 3  # Number of epochs to train with frozen backbone
UNFREEZE_LR_FACTOR = 0.1  # Learning rate factor when unfreezing

# Phase-specific learning rates
FREEZE_PHASE_LR = 1e-4    # Higher LR for classification head training
UNFREEZE_PHASE_LR = 1e-5  # Lower LR for full model fine-tuning

# =============================================================================
# VIT-SPECIFIC PATHS
# =============================================================================

# Override results directory for ViT
RESULTS_DIR = "vit_impl/models"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# Training history file
TRAINING_HISTORY_FILE = os.path.join(RESULTS_DIR, "training_history.json")
TRAINING_CURVES_FILE = os.path.join(RESULTS_DIR, "training_curves.png")
