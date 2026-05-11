import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for skin lesion classification.
    
    Expects folder structure:
    data/
      ├── melanoma/
      │   ├── image1.jpg
      │   └── ...
      ├── nevus/
      │   ├── image1.jpg
      │   └── ...
      └── seborrheic_keratosis/
          ├── image1.jpg
          └── ...
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to data folder containing class subfolders
            transform (callable, optional): Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Map class names to integers
        self.class_to_idx = {
            "melanoma": 0,
            "nevus": 1,
            "seborrheic_keratosis": 2
        }
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Scan each class folder
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Warning: Class folder not found: {class_dir}")
                continue
            
            # Collect all image files in this class
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"✓ Loaded {len(self.image_paths)} images from {root_dir}")
        
        # Print class distribution
        for class_idx, class_name in self.idx_to_class.items():
            count = sum(1 for label in self.labels if label == class_idx)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Return a single image and its label.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new("RGB", (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def get_transforms(split="train"):
    """
    Get torchvision transforms for train/val/test.

    Training uses stronger augmentations to improve generalization. Val/test stay
    deterministic (resize + normalize) to match inference.

    Args:
        split (str): "train", "val", or "test"

    Returns:
        torchvision.transforms.Compose object
    """

    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        # Stronger augments: scale/crop + affine + color + mild regularization.
        # Kept conservative for lesions (no heavy RandAugment) so structure stays plausible.
        return transforms.Compose(
            [
                # Spatial: slight scale/crop jitter (lesions often near-centered in HAM10000)
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.85, 1.0),
                    ratio=(0.92, 1.08),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.06, 0.06),
                    scale=(0.95, 1.05),
                    shear=(-5, 5),
                    fill=0,
                ),
                # Photometric
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                    p=0.25,
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.2),
                transforms.RandomAutocontrast(p=0.1),
                transforms.ToTensor(),
                # Light occlusion regularization in [0, 1] space (before ImageNet norm)
                transforms.RandomErasing(
                    p=0.12,
                    scale=(0.02, 0.06),
                    ratio=(0.3, 3.0),
                    value="random",
                ),
                imagenet_norm,
            ]
        )

    # Val / test: deterministic, aligned with typical inference (resize to 224)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing MedicalImageDataset...\n")
    
    # Check if data exists
    if not os.path.exists("data"):
        print("❌ Error: data/ folder not found")
        print("Run organize_ham10000.py first to set up the dataset")
        exit(1)
    
    # Create dataset with training transforms
    train_transforms = get_transforms(split="train")
    dataset = MedicalImageDataset(
        root_dir="data/",
        transform=train_transforms
    )
    
    print(f"\n✓ Dataset created with {len(dataset)} images")
    
    # Test loading one sample
    print("\nLoading a sample image...")
    image, label = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label} ({dataset.idx_to_class[label]})")
    print("\n✅ Dataset test passed!")
