"""
Dataset class for Pneumonia Detection using CNN
Clean modular implementation with stratified split, custom normalization, and class weights
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_config import *

class PneumoniaDataset(Dataset):
    """
    Dataset class for Pneumonia Detection from Chest X-rays
    Outputs 1-channel grayscale images with custom normalization
    """
    
    def __init__(self, image_paths, labels, split='train', custom_normalize_stats=None, augmentation_type='none'):
        """
        Initialize the dataset
        
        Args:
            image_paths (list): List of paths to image files
            labels (list): List of corresponding labels (0=NORMAL, 1=PNEUMONIA)
            split (str): Dataset split - 'train', 'test', or 'val'
            custom_normalize_stats (tuple): (mean, std) for normalization
            augmentation_type (str): 'none', 'light', or 'aggressive'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.split = split
        self.augment = (split == 'train' and augmentation_type != 'none')
        self.custom_normalize_stats = custom_normalize_stats
        self.augmentation_type = augmentation_type
        
        # Define preprocessing transforms
        self.transform = self._get_transform()
        
        print(f"✅ Created {split} dataset with {len(self.image_paths)} images")
        print(f"   - NORMAL: {sum(1 for label in self.labels if label == 0)}")
        print(f"   - PNEUMONIA: {sum(1 for label in self.labels if label == 1)}")
        print(f"   - Augmentation: {augmentation_type}")
    
    def _get_transform(self):
        """Get the complete transform pipeline based on augmentation type"""
        transform_list = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
        ]
        
        # Add augmentations for training
        if self.augment:
            if self.augmentation_type == 'aggressive':
                # RandAugment
                fill_value = int(self.custom_normalize_stats[0] * 255) if self.custom_normalize_stats else 0
                transform_list.append(
                    transforms.RandAugment(
                        magnitude=RANDAUGMENT_MAGNITUDE,
                        num_ops=RANDAUGMENT_NUM_OPS,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=fill_value
                    )
                )
            elif self.augmentation_type == 'light':
                # Light augmentations
                transform_list.extend([
                    transforms.RandomRotation(LIGHT_AUG_ROTATION),
                    transforms.RandomHorizontalFlip(LIGHT_AUG_FLIP_PROB),
                    transforms.ColorJitter(
                        brightness=LIGHT_AUG_BRIGHTNESS,
                        contrast=LIGHT_AUG_CONTRAST
                    )
                ])
        
        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        
        if self.custom_normalize_stats:
            mean, std = self.custom_normalize_stats
            transform_list.append(transforms.Normalize(mean=[mean], std=[std]))
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load and transform image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # Get label
            label = self.labels[idx]
            
            return image, label
            
        except Exception as e:
            print(f"⚠️ Error loading image {self.image_paths[idx]}: {e}")
            # Return None for corrupted images - will be filtered out by collate_fn
            return None

def collate_fn(batch):
    """Custom collate function to filter out None values from corrupted images"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_dataset_statistics(dataset, batch_size=CALCULATE_STATS_BATCH_SIZE):
    """
    Calculate mean and std for the dataset
    Returns statistics for actual data distribution
    """
    # Create a dataloader without normalization to get raw pixel stats
    temp_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # Create temporary dataset with basic transform
    if hasattr(dataset, 'transform'):
        original_transform = dataset.transform
        dataset.transform = temp_transform
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    mean = torch.zeros(1)  # Single channel for grayscale
    std = torch.zeros(1)
    total_samples = 0
    
    print("📊 Calculating dataset statistics...")
    for i, (data, _) in enumerate(loader):
        batch_samples = data.size(0)
        # Flatten H and W dimensions, keep batch and channel
        data = data.view(batch_samples, data.size(1), -1)
        
        # Calculate mean and std for this batch
        mean += data.mean(2).sum(0)  # Sum across batch
        std += data.std(2).sum(0)   # Sum across batch
        total_samples += batch_samples
    
    # Average across all samples
    mean /= total_samples
    std /= total_samples
    
    # Restore original transform if it existed
    if hasattr(dataset, 'transform') and 'original_transform' in locals():
        dataset.transform = original_transform
    
    return mean.item(), std.item()

def get_image_paths_and_labels(data_path):
    """Extract image paths and labels from ImageFolder-style directory structure"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_path, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_idx)
    
    return image_paths, labels

def create_data_loaders(augmentation_strategy='none', batch_size=BATCH_SIZE):
    """
    Create data loaders with stratified split and custom normalization
    
    Args:
        augmentation_strategy (str): 'none', 'light', or 'aggressive'
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_counts)
    """
    print(f"🔄 Creating data loaders with '{augmentation_strategy}' augmentation...")
    
    # Get dataset paths
    train_paths, train_labels = get_image_paths_and_labels(TRAIN_PATH)
    val_paths, val_labels = get_image_paths_and_labels(VAL_PATH)
    test_paths, test_labels = get_image_paths_and_labels(TEST_PATH)
    
    # Create temporary dataset for statistics calculation
    temp_dataset = datasets.ImageFolder(
        TRAIN_PATH,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    )
    
    # Calculate custom normalization statistics
    dataset_mean, dataset_std = calculate_dataset_statistics(temp_dataset)
    print(f"   Dataset statistics: mean={dataset_mean:.4f}, std={dataset_std:.4f}")
    
    # Create stratified train/val split from train+val data
    combined_paths = train_paths + val_paths
    combined_labels = train_labels + val_labels
    
    train_paths_new, val_paths_new, train_labels_new, val_labels_new = train_test_split(
        combined_paths, combined_labels,
        test_size=STRATIFIED_VAL_RATIO,
        stratify=combined_labels,
        random_state=STRATIFIED_RANDOM_STATE
    )
    
    # Create datasets
    train_dataset = PneumoniaDataset(
        train_paths_new, train_labels_new, 
        split='train', 
        custom_normalize_stats=(dataset_mean, dataset_std),
        augmentation_type=augmentation_strategy
    )
    
    val_dataset = PneumoniaDataset(
        val_paths_new, val_labels_new, 
        split='val', 
        custom_normalize_stats=(dataset_mean, dataset_std),
        augmentation_type='none'
    )
    
    test_dataset = PneumoniaDataset(
        test_paths, test_labels, 
        split='test', 
        custom_normalize_stats=(dataset_mean, dataset_std),
        augmentation_type='none'
    )
    
    # Calculate class counts for loss weighting
    train_class_counts = Counter(train_labels_new)
    normal_count = train_class_counts[0]
    pneumonia_count = train_class_counts[1]
    
    print(f"   Training set: {normal_count} NORMAL, {pneumonia_count} PNEUMONIA")
    print(f"   Class ratio: {normal_count/pneumonia_count:.1f}:1")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, (normal_count, pneumonia_count)