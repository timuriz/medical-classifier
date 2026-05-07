#!/usr/bin/env python3
"""
Training Script for Skin Lesion Classifier

This script:
1. Loads the HAM10000 dataset
2. Fine-tunes EfficientNet-B0 on skin lesion images
3. Saves the best model weights
4. Evaluates on test set

Requirements:
  pip install torch torchvision torch-optim pillow tqdm
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import json
from datetime import datetime

# Import dataset class
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.dataset import MedicalImageDataset, get_transforms

# Configuration
CONFIG = {
    "data_dir": "data/",
    "models_dir": "models/",
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "val_split": 0.2,
    "test_split": 0.1,
    "num_workers": 0,
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def create_model(num_classes=3):
    """Create EfficientNet-B0 model for skin lesion classification."""
    print("\n📦 Loading EfficientNet-B0 (pretrained on ImageNet)...")
    
    model = efficientnet_b0(pretrained=True)
    
    # Replace final layer for 3-class classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Freeze early layers (transfer learning)
    print("   Freezing early layers, training only final layer...")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": total_loss / (total / CONFIG["batch_size"])})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, epoch, num_epochs):
    """Validate on validation set."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("SKIN LESION CLASSIFIER - TRAINING")
    print("=" * 70)
    
    # Create models directory
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    
    # Load dataset
    print(f"\n📁 Loading dataset from {CONFIG['data_dir']}...")
    dataset = MedicalImageDataset(
        root_dir=CONFIG["data_dir"],
        transform=None  # Will apply transforms in DataLoader
    )
    
    # Split into train/val/test
    total_size = len(dataset)
    test_size = int(total_size * CONFIG["test_split"])
    val_size = int((total_size - test_size) * CONFIG["val_split"])
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    print(f"\n📊 Split sizes:")
    print(f"   Training:   {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    print(f"   Test:       {len(test_dataset)} images")
    
    # Create DataLoaders with augmentation
    train_transforms = get_transforms(split="train")
    val_transforms = get_transforms(split="val")
    
    # Wrap datasets with transforms
    class TransformDataset:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            image = self.transform(image)
            return image, label
    
    train_dataset = TransformDataset(train_dataset, train_transforms)
    val_dataset = TransformDataset(val_dataset, val_transforms)
    test_dataset = TransformDataset(test_dataset, val_transforms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    
    # Create model
    model = create_model(num_classes=3)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...\n")
    
    best_val_acc = 0.0
    best_model_path = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(CONFIG["num_epochs"]):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, CONFIG["num_epochs"])
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, CONFIG["num_epochs"])
        
        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"\n  Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CONFIG["models_dir"], "efficientnet_isic_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved: {best_model_path}")
    
    # Test set evaluation
    print(f"\n{'='*70}")
    print("TESTING ON TEST SET")
    print(f"{'='*70}\n")
    
    model.load_state_dict(torch.load(best_model_path))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    
    test_loss, test_acc = validate(model, test_loader, criterion, 0, 1)
    print(f"\n  Test: loss={test_loss:.4f}, acc={test_acc:.1f}%")
    
    # Save training history
    history_file = os.path.join(CONFIG["models_dir"], "training_history.json")
    with open(history_file, "w") as f:
        json.dump({
            "config": CONFIG,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {best_model_path}")
    print(f"   History: {history_file}")
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc:.1f}%")
    print(f"   Best Val Accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    main()
