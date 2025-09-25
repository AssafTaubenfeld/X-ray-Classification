"""
CNN-specific Configuration for Pneumonia Detection
Contains parameters specific to CNN implementation
"""
from base_config import *
from enum import Enum

# =============================================================================
# CNN ARCHITECTURE CONFIGURATION
# =============================================================================

class CNNArchitecture(Enum):
    """Enum for different CNN architecture variants"""
    TWO_CONV_LAYERS = "2 layers"
    THREE_CONV_LAYERS = "3 layers"
    FOUR_CONV_LAYERS = "4 layers"

# Default architecture
DEFAULT_ARCHITECTURE = CNNArchitecture.THREE_CONV_LAYERS

# =============================================================================
# CNN MODEL PARAMETERS
# =============================================================================

# Architecture-specific parameters
ADAPTIVE_POOL_SIZE = (4, 4)  # Adaptive pooling output size
DROPOUT_RATE = 0.5           # Dropout rate in classification head

# Classification head configuration
CLASSIFIER_HIDDEN_SIZE = 256  # Hidden layer size in classifier
NUM_CLASSES_OUTPUT = 1        # Output size (1 for binary classification with sigmoid)

# =============================================================================
# CNN TRAINING HYPERPARAMETERS
# =============================================================================

# Learning rate for CNN training (higher than ViT since training from scratch)
LEARNING_RATE = 1e-3

# CNN-specific batch size
BATCH_SIZE = 32

# =============================================================================
# CNN-SPECIFIC PATHS
# =============================================================================

# Override results directory for CNN
RESULTS_DIR = "cnn_impl/models"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# =============================================================================
# ARCHITECTURE COMPARISON CONFIGURATION
# =============================================================================

# For architecture comparison experiments
ARCHITECTURE_COMPARISON_CONFIGS = {
    CNNArchitecture.TWO_CONV_LAYERS: {
        'name': '2_Layers',
        'results_subdir': '2_layers_model'
    },
    CNNArchitecture.THREE_CONV_LAYERS: {
        'name': '3_Layers', 
        'results_subdir': '3_layers_model'
    },
    CNNArchitecture.FOUR_CONV_LAYERS: {
        'name': '4_Layers',
        'results_subdir': '4_layers_model'
    }
}

