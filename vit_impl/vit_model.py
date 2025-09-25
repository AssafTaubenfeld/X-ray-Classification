"""
Vision Transformer (ViT) Model for Pneumonia Detection
Adapts pre-trained ViT for grayscale medical imaging with proper weight initialization
Supports freeze-then-unfreeze training strategy for optimal transfer learning
"""

import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vit_config import *

warnings.filterwarnings('ignore')

class PneumoniaViT(nn.Module):
    """
    Vision Transformer for Pneumonia Detection from Chest X-rays
    
    Key Modifications: 
    1. Adapts pre-trained ViT to accept 1-channel grayscale input
    2. Initializes grayscale weights by averaging RGB weights
    3. Updates classifier head for binary classification
    4. Supports freeze-then-unfreeze training strategy
    5. Maintains all pre-trained features and attention mechanisms
    """
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, freeze_backbone=False):
        """
        Initialize the ViT model for pneumonia detection
        
        Args:
            model_name (str): Pre-trained ViT model name from Hugging Face
            num_classes (int): Number of output classes (2 for NORMAL/PNEUMONIA)
            freeze_backbone (bool): Whether to freeze the backbone initially
        """
        super(PneumoniaViT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.initial_freeze_backbone = freeze_backbone
        
        # Track freeze state for freeze-then-unfreeze strategy
        self.is_backbone_frozen = False
        self.freeze_unfreeze_history = []  # Track transitions for logging
        
        print(f"🚀 Loading pre-trained ViT model: {model_name}")
        
        # Load pre-trained ViT model
        self._load_pretrained_model()
        
        # Modify for grayscale input (1-channel instead of 3-channel)
        self._adapt_for_grayscale_input()
        
        # Optionally freeze backbone initially
        if freeze_backbone:
            self.freeze_backbone()
        
        print("✅ ViT model successfully adapted for pneumonia detection!")
        self._print_model_info()
    
    def _load_pretrained_model(self):
        """Load the pre-trained ViT model from Hugging Face"""
        try:
            config = ViTConfig.from_pretrained(self.model_name)
            config.num_channels = 1
            config.num_labels = self.num_classes

            self.vit = ViTForImageClassification.from_pretrained(
                self.model_name,
                config=config,
                ignore_mismatched_sizes=True  # Allow different number of classes
            )
            print(f"✅ Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        print(f"✅ Classifier head updated: {self.vit.classifier.in_features} -> {self.vit.classifier.out_features}")
    
    def _adapt_for_grayscale_input(self):
        """
        Modify the patch embedding layer to accept 1-channel grayscale input
        Initialize new weights by averaging the pre-trained RGB weights
        """
        print("🔧 Adapting model for grayscale input...")
        
        # Get the original patch embedding layer
        original_conv = self.vit.vit.embeddings.patch_embeddings.projection
        
        print(f"   Original input channels: {original_conv.in_channels}")
        print(f"   Target input channels: 1 (grayscale)")
        
        # Create new Conv2d layer for grayscale input
        new_conv = nn.Conv2d(
            in_channels=1,  # Grayscale input
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights by averaging RGB channels
        with torch.no_grad():
            # Average the weights across the RGB channels (dim=1)
            averaged_weights = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(averaged_weights)
            
            # Copy bias if it exists
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        # Replace the original layer
        self.vit.vit.embeddings.patch_embeddings.projection = new_conv
        
        print("✅ Successfully adapted patch embeddings for grayscale input")
        print(f"   New patch embedding shape: {new_conv.weight.shape}")
    
    def freeze_backbone(self, log_transition=True):
        """
        Freeze the backbone layers (keep only classifier trainable)
        
        Args:
            log_transition (bool): Whether to log this transition
        """
        if self.is_backbone_frozen:
            if log_transition:
                print("ℹ️  Backbone is already frozen")
            return
        
        if log_transition:
            print("🧊 Freezing backbone layers...")
        
        # Freeze all parameters except classifier
        for name, param in self.vit.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        self.is_backbone_frozen = True
        
        # Log transition
        self.freeze_unfreeze_history.append("frozen")
        
        if log_transition:
            frozen_params = sum(p.numel() for p in self.vit.parameters() if not p.requires_grad)
            trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            
            print(f"✅ Backbone frozen:")
            print(f"   Frozen parameters: {frozen_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
    
    def unfreeze_backbone(self, log_transition=True):
        """
        Unfreeze the backbone layers (make entire model trainable)
        
        Args:
            log_transition (bool): Whether to log this transition
        """
        if not self.is_backbone_frozen:
            if log_transition:
                print("ℹ️  Backbone is already unfrozen")
            return
        
        if log_transition:
            print("🔥 Unfreezing backbone layers...")
        
        # Unfreeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = True
        
        self.is_backbone_frozen = False
        
        # Log transition
        self.freeze_unfreeze_history.append("unfrozen")
        
        if log_transition:
            trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            
            print(f"✅ Backbone unfrozen:")
            print(f"   All parameters trainable: {trainable_params:,}")
    
    def is_frozen(self):
        """Check if backbone is currently frozen"""
        return self.is_backbone_frozen
    
    def get_freeze_history(self):
        """Get the history of freeze/unfreeze transitions"""
        return self.freeze_unfreeze_history.copy()
    
    def print_freeze_status(self):
        """Print current freeze status and parameter counts"""
        total_params = sum(p.numel() for p in self.vit.parameters())
        trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        status = "FROZEN" if self.is_backbone_frozen else "UNFROZEN"
        
        print(f"🔍 Current Model Status: {status}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {frozen_params:,}")
        print(f"   Freeze history: {self.freeze_unfreeze_history}")
    
    def setup_freeze_then_unfreeze(self, freeze_epochs, total_epochs):
        """
        Setup information for freeze-then-unfreeze training strategy
        
        Args:
            freeze_epochs (int): Number of epochs to train with frozen backbone
            total_epochs (int): Total number of training epochs
            
        Returns:
            dict: Training strategy information
        """
        if freeze_epochs >= total_epochs:
            raise ValueError(f"freeze_epochs ({freeze_epochs}) must be less than total_epochs ({total_epochs})")
        
        strategy_info = {
            'freeze_epochs': freeze_epochs,
            'unfreeze_epochs': total_epochs - freeze_epochs,
            'total_epochs': total_epochs,
            'unfreeze_at_epoch': freeze_epochs,
            'strategy': 'freeze-then-unfreeze'
        }
        
        print(f"📋 Freeze-then-Unfreeze Strategy:")
        print(f"   Phase 1 (Frozen): Epochs 0-{freeze_epochs-1} ({freeze_epochs} epochs)")
        print(f"   Phase 2 (Unfrozen): Epochs {freeze_epochs}-{total_epochs-1} ({total_epochs-freeze_epochs} epochs)")
        print(f"   Unfreeze transition at epoch: {freeze_epochs}")
        
        return strategy_info
    
    def transition_at_epoch(self, current_epoch, strategy_info):
        """
        Handle freeze/unfreeze transition at specified epoch
        
        Args:
            current_epoch (int): Current training epoch (0-indexed)
            strategy_info (dict): Strategy info from setup_freeze_then_unfreeze
            
        Returns:
            bool: True if transition occurred, False otherwise
        """
        if current_epoch == strategy_info['unfreeze_at_epoch']:
            print(f"\n🔄 EPOCH {current_epoch}: Transitioning to unfrozen training")
            print("=" * 60)
            self.unfreeze_backbone()
            print("=" * 60)
            print("💡 Tip: Consider reducing learning rate for fine-tuning phase")
            return True
        return False
    
    def _print_model_info(self):
        """Print detailed model information"""
        total_params = sum(p.numel() for p in self.vit.parameters())
        trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Information:")
        print(f"   Model: {self.model_name}")
        print(f"   Input: 1-channel grayscale images ({IMAGE_SIZE}x{IMAGE_SIZE})")
        print(f"   Output: {self.num_classes} classes (NORMAL, PNEUMONIA)")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")
        print(f"   Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
        print(f"   Backbone status: {'FROZEN' if self.is_backbone_frozen else 'UNFROZEN'}")
        
        # Print input/output shapes
        print(f"\n🔧 Expected Shapes:")
        print(f"   Input: (batch_size, 1, {IMAGE_SIZE}, {IMAGE_SIZE})")
        print(f"   Output: (batch_size, {self.num_classes})")
    
    def forward(self, images):
        """
        Forward pass through the ViT model
        
        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """

        # Ensure input has correct shape
        if images.dim() != 4 or images.size(1) != 1:
            raise ValueError(f"Expected input shape (batch_size, 1, H, W), got {images.shape}")
        
        # Forward pass through ViT
        outputs = self.vit(pixel_values=images)
        
        return outputs.logits
    
    def get_attention_weights(self, pixel_values, layer_idx=-1):
        """
        Extract attention weights from a specific layer
        Useful for visualization and interpretability
        
        Args:
            pixel_values (torch.Tensor): Input tensor
            layer_idx (int): Which transformer layer to extract attention from (-1 for last)
            
        Returns:
            torch.Tensor: Attention weights
        """
        self.vit.eval()
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
            attention_weights = outputs.attentions[layer_idx]
        
        return attention_weights
    
    def get_feature_embeddings(self, pixel_values):
        """
        Extract feature embeddings before classification
        Useful for feature analysis and similarity computation
        
        Args:
            pixel_values (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature embeddings
        """
        self.vit.eval()
        with torch.no_grad():
            # Get hidden states from the last layer
            outputs = self.vit.vit(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state
            
            # Use CLS token embedding (first token)
            cls_embedding = last_hidden_state[:, 0, :]
            
        return cls_embedding

def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, freeze_backbone=False):
    """
    Factory function to create a ViT model for pneumonia detection
    
    Args:
        model_name (str): Pre-trained model name
        num_classes (int): Number of output classes
        freeze_backbone (bool): Whether to freeze backbone initially
        
    Returns:
        PneumoniaViT: Configured model ready for training
    """
    print("🏭 Creating ViT model for pneumonia detection...")
    
    model = PneumoniaViT(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )
    
    return model

def test_model():
    """Test the model creation and basic functionality"""
    print("🧪 Testing ViT model...")
    
    # Create model
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)
    
    print(f"\n🔬 Testing forward pass:")
    print(f"   Input shape: {test_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test attention extraction
    print(f"\n👁️  Testing attention extraction:")
    attention = model.get_attention_weights(test_input)
    print(f"   Attention shape: {attention.shape}")
    
    # Test feature extraction
    print(f"\n🧩 Testing feature extraction:")
    features = model.get_feature_embeddings(test_input)
    print(f"   Feature embedding shape: {features.shape}")
    
    print("✅ All model tests passed!")
    
    return model

if __name__ == "__main__":
    # Test basic model functionality
    model = test_model()
    
    print("\n" + "="*80)
    
    # Print final model summary
    print(f"\n📋 Final Model Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
