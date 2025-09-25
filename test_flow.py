import torch
import os
import json
import seaborn as sns
from datetime import datetime
import numpy as np

from train import calculate_binary_metrics

from utils import setup_environment

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cnn_config import *

def evaluate_on_test_set(model, test_loader, device):
    """
    Evaluate the best model on test set and generate comprehensive reports
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        results_dir: Directory to save results
        
    Returns:
        dict: Test metrics
    """
    print("\n🧪 Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probabilities.extend(probabilities.cpu().numpy().flatten())
    
    test_metrics = calculate_binary_metrics(
        np.array(all_labels), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    print(f"📊 Test Results:")
    print(f"   • Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   • Precision: {test_metrics['precision']:.4f}")
    print(f"   • Recall/Sensitivity: {test_metrics['recall']:.4f}")
    print(f"   • Specificity: {test_metrics['specificity']:.4f}")
    print(f"   • F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   • AUC-ROC: {test_metrics['auc']:.4f}")
    
    
    return test_metrics

def test_model(model, test_loader, results_dir, checkpoint_dir):
    """
    Test the trained model on test set and generate comprehensive reports
    
    Args:
        model: Trained model object
        test_loader: DataLoader for test data
        
    Returns:
        dict: Test metrics and results
    """
    print("🧪 Starting CNN Model Testing...")
    
    # Setup environment (mainly for device)
    device = setup_environment(results_dir, checkpoint_dir)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Evaluate on test set
    test_metrics = evaluate_on_test_set(model, test_loader, device)
    
    # Save final test results
    final_test_results = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'], 
        'test_recall': test_metrics['recall'],
        'test_f1_score': test_metrics['f1_score'],
        'test_auc': test_metrics['auc'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'test_samples': len(test_loader.dataset),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(results_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_test_results, f, indent=2)
    
    print(f"\n🎯 Final Test Performance:")
    print(f"   • Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   • Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   • Test AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"   • Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"   • Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"📝 Test results saved to: {results_path}")
    
    return final_test_results