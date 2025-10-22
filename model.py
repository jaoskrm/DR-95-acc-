# ================================================================
# ENHANCED VERSION WITH PER-EPOCH CONFUSION MATRIX
# ================================================================

import os, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (cohen_kappa_score, f1_score, 
                            classification_report, confusion_matrix)
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_PATH = Path(r"D:\studies\clg\cao\data\colored_images")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NEW: Display options
SHOW_CONFUSION_MATRIX = True  # Set to False to hide per-epoch CM
SHOW_PER_CLASS_METRICS = True  # Show precision/recall per class

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_per_process_memory_fraction(0.90)
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  WARNING: GPU recommended for faster training")
    BATCH_SIZE = 8

# ==================== ENHANCED PREPROCESSING ====================

def enhance_fundus_image(image):
    """Light preprocessing - images are already good quality"""
    try:
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    except:
        return image

# ==================== DATASET ====================

class DRDataset(Dataset):
    def __init__(self, image_paths, labels, transform, apply_enhancement=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_enhancement = apply_enhancement
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            
            if self.apply_enhancement:
                image = enhance_fundus_image(image)
            
            image = np.array(image)
            image = self.transform(image=image)['image']
            label = self.labels[idx]
            
            return image, label
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), self.labels[idx]

# ==================== TRANSFORMS ====================

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==================== MODEL ====================

class EfficientNetDR(nn.Module):
    """EfficientNet-B3 for DR classification - optimal for 224x224"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class ResNetDR(nn.Module):
    """ResNet50 for DR classification"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==================== LOSS FUNCTIONS ====================

class QWKLoss(nn.Module):
    """Differentiable QWK loss for ordinal classification"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        w = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = (i - j)**2 / (num_classes - 1)**2
        self.w = torch.from_numpy(w).float().to(DEVICE)

    def forward(self, logits, targets):
        targets = targets.long()
        probs = torch.softmax(logits, dim=1)
        O = torch.matmul(probs.T, F.one_hot(targets, self.num_classes).float())
        O = O / (O.sum() + 1e-8)
        pred_hist = probs.sum(0, keepdim=True)
        true_hist = F.one_hot(targets, self.num_classes).float().sum(0, keepdim=True)
        E = torch.matmul(true_hist.T, pred_hist)
        E = E / (E.sum() + 1e-8)
        num = (self.w * O).sum()
        den = (self.w * E).sum()
        return 1 - (1 - num) / (1 - den + 1e-8)


class CombinedLoss(nn.Module):
    """Combines Cross-Entropy with QWK loss"""
    def __init__(self, num_classes=5, alpha=0.7):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.qwk_loss = QWKLoss(num_classes)
        self.alpha = alpha
        
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        qwk = self.qwk_loss(logits, targets)
        return self.alpha * ce + (1 - self.alpha) * qwk

# ==================== DATA LOADING ====================

def load_data(data_path):
    """Load DR dataset from folder structure"""
    folder_to_grade = {
        'No_DR': 0,
        'Mild': 1,
        'Moderate': 2,
        'Severe': 3,
        'Proliferate_DR': 4
    }
    
    all_paths = []
    all_labels = []
    
    print("\nLoading dataset...")
    for folder_name, grade in folder_to_grade.items():
        class_dir = data_path / folder_name
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} not found")
            continue
        
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            images.extend(list(class_dir.glob(ext)))
        
        all_paths.extend(images)
        all_labels.extend([grade] * len(images))
        
        print(f"{folder_name} (Grade {grade}): {len(images)} images")
    
    print(f"\n✓ Total images: {len(all_paths)}")
    
    if len(all_paths) == 0:
        print(f"\n❌ ERROR: No images found in {data_path}")
        exit(1)
    
    return all_paths, all_labels, list(folder_to_grade.keys())

# ==================== ENHANCED DISPLAY FUNCTIONS ====================

def print_confusion_matrix(cm, class_names):
    """Print formatted confusion matrix"""
    print("\n📊 Confusion Matrix:")
    
    # Header
    header = "        " + "".join([f"{name:>10s}" for name in class_names])
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>8s}"
        for val in row:
            row_str += f"{val:>10d}"
        print(row_str)
    print()


def print_per_class_metrics(y_true, y_pred, class_names):
    """Print per-class precision, recall, F1"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    print("📈 Per-Class Metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 65)
    
    for i, name in enumerate(class_names):
        print(f"{name:<15} {precision[i]:>10.3f} {recall[i]:>10.3f} {f1[i]:>10.3f} {support[i]:>10d}")
    
    print("-" * 65)
    print(f"{'Macro Avg':<15} {precision.mean():>10.3f} {recall.mean():>10.3f} {f1.mean():>10.3f} {support.sum():>10d}")
    print()

# ==================== TRAINING & EVALUATION ====================

def train_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, class_names, show_details=True):
    """ENHANCED: Now shows confusion matrix and per-class metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = np.mean(all_labels == all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Display detailed metrics if requested
    if show_details:
        if SHOW_CONFUSION_MATRIX:
            print_confusion_matrix(cm, class_names)
        
        if SHOW_PER_CLASS_METRICS:
            print_per_class_metrics(all_labels, all_preds, class_names)
    
    return {'qwk': qwk, 'f1': f1, 'accuracy': acc, 'confusion_matrix': cm}, all_preds, all_labels

# ==================== MAIN TRAINING ====================

def main():
    # Load data
    all_paths, all_labels, class_names = load_data(DATA_PATH)
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create datasets
    train_ds = DRDataset(train_paths, train_labels, train_transform, apply_enhancement=True)
    val_ds = DRDataset(val_paths, val_labels, val_transform, apply_enhancement=False)
    test_ds = DRDataset(test_paths, test_labels, val_transform, apply_enhancement=False)
    
    # DataLoaders
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {class_names}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    
    # Initialize model
    if DEVICE == 'cuda':
        print("\n✓ Using EfficientNet-B3 (optimal for 224x224)")
        model = EfficientNetDR(num_classes=5).to(DEVICE)
    else:
        print("\n⚠️  Using ResNet50 (lighter for CPU)")
        model = ResNetDR(num_classes=5).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Training setup
    criterion = CombinedLoss(num_classes=5, alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    best_qwk = 0
    patience = 15
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        
        # Validate with detailed metrics
        metrics, val_preds, val_labels_arr = validate_epoch(
            model, val_loader, class_names, show_details=True
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"📊 EPOCH {epoch+1} SUMMARY")
        print(f"{'─'*60}")
        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Val QWK:        {metrics['qwk']:.4f}")
        print(f"Val F1:         {metrics['f1']:.4f}")
        print(f"Val Accuracy:   {metrics['accuracy']*100:.2f}%")
        print(f"Learning Rate:  {current_lr:.2e}")
        
        # Save best model
        if metrics['qwk'] > best_qwk:
            best_qwk = metrics['qwk']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_qwk': best_qwk,
                'metrics': metrics,
                'confusion_matrix': metrics['confusion_matrix']
            }, DATA_PATH.parent / 'best_dr_model.pth')
            
            print(f"✅ NEW BEST MODEL! QWK: {best_qwk:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(DATA_PATH.parent / 'best_dr_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_preds, test_labels_arr = validate_epoch(
        model, test_loader, class_names, show_details=True
    )
    
    print(f"\n🎯 FINAL TEST RESULTS")
    print(f"{'─'*60}")
    print(f"Test QWK:       {test_metrics['qwk']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Best Validation QWK: {best_qwk:.4f}")
    print(f"Final Test QWK:      {test_metrics['qwk']:.4f}")
    print(f"Model saved to:      {DATA_PATH.parent / 'best_dr_model.pth'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    main()
