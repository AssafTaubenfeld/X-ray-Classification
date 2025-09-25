"""
CNN Model for Pneumonia Detection from Chest X-rays
Clean modular implementation with configurable architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cnn_config import *

class PneumoniaCNN(nn.Module):
    """
    CNN for Pneumonia Detection from Chest X-rays
    
    Architecture:
    - Feature Extraction Base: Configurable convolutional blocks based on architecture type
    - Classification Head: Fully connected layers with dropout
    """
    
    def __init__(self, num_classes=NUM_CLASSES_OUTPUT, 
                 adaptive_pool_size=ADAPTIVE_POOL_SIZE, 
                 architecture=DEFAULT_ARCHITECTURE,
                 dropout_rate=DROPOUT_RATE,
                 classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE):
        super(PneumoniaCNN, self).__init__()
        
        self.architecture = architecture
        
        # Build the feature extraction backbone
        if architecture == CNNArchitecture.THREE_CONV_LAYERS:
            self._build_three_block_architecture()
        elif architecture == CNNArchitecture.TWO_CONV_LAYERS:
            self._build_two_block_architecture()
        elif architecture == CNNArchitecture.FOUR_CONV_LAYERS:
            self._build_four_block_architecture()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
        # Adaptive Pooling - Makes model input-size agnostic
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        
        # Calculate flatten size dynamically based on adaptive pool output and final channels
        self.flatten_size = self.final_channels * adaptive_pool_size[0] * adaptive_pool_size[1]
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, classifier_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_two_block_architecture(self):
        """Build the basic 2-block architecture"""
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
  
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.final_channels = 32
        self.blocks = [self.block1, self.block2]
    
    def _build_three_block_architecture(self):
        """Build the basic 3-block architecture"""
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
  
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.final_channels = 64
        self.blocks = [self.block1, self.block2, self.block3]
    
    def _build_four_block_architecture(self):
        """Build the deeper 4-block architecture with more channels"""
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
  
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.final_channels = 128
        self.blocks = [self.block1, self.block2, self.block3, self.block4]

    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction through all blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.adaptive_pool(x)

        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extract feature maps from each block for visualization
        Returns: Dictionary with feature maps from each block
        """
        features = {}
        
        for i, block in enumerate(self.blocks, 1):
            x = block(x)
            features[f'block{i}'] = x.clone()
        
        return features

def create_model(device='cpu', architecture=DEFAULT_ARCHITECTURE):
    """
    Create and initialize the pneumonia detection model
    
    Args:
        device: torch device to move the model to
        architecture: CNNArchitecture enum value for model architecture
        
    Returns:
        model: Initialized PneumoniaCNN model
    """
    print(f"🚀 Creating CNN model with {architecture.value} architecture...")
    
    model = PneumoniaCNN(num_classes=NUM_CLASSES_OUTPUT, architecture=architecture)
    model = model.to(device)
    
    # Print model summary
    total_params = count_parameters(model)
    print(f"   Total trainable parameters: {total_params:,}")
    
    return model

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, input_size=(1, 1, IMAGE_SIZE, IMAGE_SIZE)):
    """
    Print detailed model architecture summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
    """
    print("📋 Model Architecture Summary:")
    print("=" * 60)
    
    total_params = count_parameters(model)
    print(f"Architecture: {model.architecture.value}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Input size: {input_size}")
    print()
    
    # Print layer details
    print("Layer Details:")
    print("-" * 60)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<30} {str(module):<40} {params:>10,} params")
    
    print("=" * 60)
