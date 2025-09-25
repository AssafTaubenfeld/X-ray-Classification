"""
Base Configuration for Pneumonia Detection Project
Contains shared parameters between CNN and ViT implementations
"""
import os
import kagglehub

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Dataset path
DATA_PATH = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Dataset subdirectories
TRAIN_PATH = os.path.join(DATA_PATH, "chest_xray", "train")
TEST_PATH = os.path.join(DATA_PATH, "chest_xray", "test")
VAL_PATH = os.path.join(DATA_PATH, "chest_xray", "val")

# Class names for binary classification
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# Stratified split configuration (always used)
STRATIFIED_VAL_RATIO = 0.2   # 80-20 split for better validation size
STRATIFIED_RANDOM_STATE = 42  # For reproducibility

# Custom normalization settings (always used)
CALCULATE_STATS_BATCH_SIZE = 64  # Batch size for statistics calculation

IMAGE_SIZE = 224  # Input image resolution (224x224)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Learning rate for training
LEARNING_RATE = 1e-3  # Base learning rate (will be overridden by architecture configs)

# Batch size
BATCH_SIZE = 16

# Number of training epochs
EPOCHS = 50

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Minimum improvement threshold for best model updates
MIN_DELTA = 1e-4  # Minimum improvement in validation metric to consider as improvement


# =============================================================================
# OPTIMIZATION AND REGULARIZATION
# =============================================================================

# Weight decay for regularization
WEIGHT_DECAY = 1e-4

# Learning rate scheduler settings
LR_SCHEDULER_STEP_SIZE = 3
LR_SCHEDULER_GAMMA = 0.1
LR_SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau
LR_SCHEDULER_PATIENCE = 5  # For ReduceLROnPlateau
MIN_LR = 1e-7  # Minimum learning rate

# Gradient clipping
GRAD_CLIP_VALUE = 1.0

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

# Data augmentation parameters for training
TRAIN_TRANSFORMS = {
    'rotation_degrees': 10,
    'horizontal_flip_prob': 0.5,
    'brightness': 0.2,
    'contrast': 0.2,
}

# =============================================================================
# ENVIRONMENT AND PERFORMANCE
# =============================================================================

# Number of workers for data loading
NUM_WORKERS = 4

# Pin memory for faster GPU transfer
PIN_MEMORY = True

# =============================================================================
# LOGGING AND CHECKPOINTS
# =============================================================================

# Directory to save model checkpoints and results
RESULTS_DIR = "models"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# Save frequency
SAVE_FREQ = 2

# Whether to save best model based on validation accuracy
SAVE_BEST_MODEL = True

# Logging interval during training
LOG_INTERVAL = 100

# =============================================================================
# EVALUATION
# =============================================================================

# Metrics to track during training
METRICS = ["accuracy", "precision", "recall", "f1_score", "auc"]

# Whether to generate reports and visualizations
GENERATE_REPORTS = True
SAVE_VISUALIZATIONS = True

# =============================================================================
# DATA AUGMENTATION TYPES
# =============================================================================

# Light augmentation parameters
LIGHT_AUG_ROTATION = 10
LIGHT_AUG_FLIP_PROB = 0.5
LIGHT_AUG_BRIGHTNESS = 0.2
LIGHT_AUG_CONTRAST = 0.2

# Aggressive augmentation (RandAugment) parameters
USE_RANDAUGMENT = False  # Toggle for RandAugment
RANDAUGMENT_MAGNITUDE = 9
RANDAUGMENT_NUM_OPS = 10

# =============================================================================
# DATA Processing
# =============================================================================

# Class weighting for imbalanced dataset
USE_CLASS_WEIGHTS = True
# These will be calculated dynamically based on dataset, but default values:
DEFAULT_NORMAL_COUNT = 1
DEFAULT_PNEUMONIA_COUNT = 1