# ================================================================
# PROPERLY FIXED OPTIMIZED DR TRAINING - SAM + SWA + CORAL + XAI
# Fixed: SAM optimizer workflow without breaking AMP
# Fixed: CORAL loss numerical stability
# Fixed: Model initialization and training issues
# ================================================================

import os, warnings, math, random
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
from torch.optim.swa_utils import AveragedModel, SWALR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             classification_report, confusion_matrix)
from tqdm.auto import tqdm

# XAI IMPORTS
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_PATH = Path(r"D:\studies\clg\cao\data\colored_images")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 150
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# OPTIMIZATION SETTINGS
USE_SAM = True
USE_SWA = True
SWA_START_EPOCH = 120
SWA_LR = 1e-5

USE_MIXUP = True
USE_CUTMIX = True
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5

# LOSS SETTINGS
USE_CORAL = True
USE_CLASS_BALANCED = True
CLASS_BALANCED_BETA = 0.99999

# MANUAL WEIGHT BOOSTS FOR CRITICAL CLASSES
SEVERE_WEIGHT_BOOST = 1.5
PROLIFERATIVE_WEIGHT_BOOST = 1.3

# Display options
SHOW_CONFUSION_MATRIX = True
SHOW_PER_CLASS_METRICS = True

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_per_process_memory_fraction(0.90)
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️ WARNING: GPU recommended")
    BATCH_SIZE = 8

# ==================== PREPROCESSING ====================
def enhance_fundus_image(image):
    """CLAHE preprocessing"""
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

# ==================== MIXUP / CUTMIX ====================
def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    _, _, H, W = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

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

test_transform = val_transform

# ==================== CORAL HEAD ====================
class CoralLayer(nn.Module):
    """CORAL ordinal regression output layer with numerical stability"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.coral_weights = nn.Linear(512, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1).float())
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.coral_weights.weight)
        nn.init.constant_(self.coral_bias, 0.0)

    def forward(self, x):
        return self.coral_weights(x) + self.coral_bias

# ==================== MODEL ====================
class EfficientNetDR_CORAL(nn.Module):
    """EfficientNet-B3 with CORAL ordinal regression"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        num_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()

        self.feature_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        if USE_CORAL:
            self.ordinal_head = CoralLayer(num_classes)
        else:
            self.classifier = nn.Linear(512, num_classes)
            
        # Initialize feature head properly
        for m in self.feature_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_head(features)

        if USE_CORAL:
            return self.ordinal_head(features)
        else:
            return self.classifier(features)

# ==================== CORAL UTILITIES (FIXED) ====================
def coral_loss(logits, levels):
    """CORAL ordinal regression loss with numerical stability"""
    batch_size = logits.size(0)
    num_classes = logits.size(1) + 1

    levels_expanded = levels.unsqueeze(1).repeat(1, num_classes - 1)
    thresholds = torch.arange(num_classes - 1, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(logits.device)

    targets = (levels_expanded > thresholds).float()

    # Clamp logits to prevent numerical instability
    logits_clamped = torch.clamp(logits, min=-10, max=10)
    
    loss = F.binary_cross_entropy_with_logits(logits_clamped, targets, reduction='mean')

    return loss

def coral_predict(logits):
    """Convert CORAL logits to class predictions"""
    # Clamp logits for stability
    logits_clamped = torch.clamp(logits, min=-10, max=10)
    probas = torch.sigmoid(logits_clamped)
    predictions = (probas > 0.5).sum(dim=1)
    return predictions

def coral_to_probs(logits):
    """Convert CORAL logits to class probabilities with numerical stability"""
    # Clamp logits for stability
    logits_clamped = torch.clamp(logits, min=-10, max=10)
    probas = torch.sigmoid(logits_clamped)
    
    # Add boundary probabilities
    probas_padded = torch.cat([
        torch.zeros(probas.size(0), 1, device=probas.device),
        probas,
        torch.ones(probas.size(0), 1, device=probas.device)
    ], dim=1)

    # Compute class probabilities
    class_probs = probas_padded[:, 1:] - probas_padded[:, :-1]
    
    # Ensure probabilities are non-negative and sum to 1
    class_probs = torch.clamp(class_probs, min=1e-8)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)

    return class_probs

# ==================== CLASS-BALANCED LOSS ====================
def get_effective_num(samples_per_class, beta=0.9999):
    """Calculate effective number of samples per class"""
    effective_num = 1.0 - np.power(beta, samples_per_class)
    return effective_num

def get_class_weights(samples_per_class, beta=0.99999):
    """Calculate class-balanced weights"""
    effective_num = get_effective_num(samples_per_class, beta)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)
    return torch.FloatTensor(weights)

# ==================== COMBINED LOSS (FIXED) ====================
class CombinedOrdinalLoss(nn.Module):
    """CORAL loss + QWK loss with class balancing - numerically stable"""
    def __init__(self, num_classes=5, class_weights=None, alpha=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_weights = class_weights

        w = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = (i - j)**2 / (num_classes - 1)**2
        self.w = torch.from_numpy(w).float()

    def forward(self, logits, targets):
        targets = targets.long()
        device = logits.device
        
        # Move weight matrix to device if needed
        if self.w.device != device:
            self.w = self.w.to(device)

        # CORAL loss
        coral_loss_value = coral_loss(logits, targets)

        # Apply class weights if provided
        if self.class_weights is not None:
            if self.class_weights.device != device:
                self.class_weights = self.class_weights.to(device)
            weights_batch = self.class_weights[targets]
            coral_loss_value = coral_loss_value * weights_batch.mean()

        # QWK loss with numerical stability
        try:
            probs = coral_to_probs(logits)
            
            # Ensure probs and targets are valid
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Warning: NaN/Inf in probabilities, using CORAL loss only")
                return coral_loss_value
            
            O = torch.matmul(probs.T, F.one_hot(targets, self.num_classes).float())
            O = O / (O.sum() + 1e-8)
            
            pred_hist = probs.sum(0, keepdim=True)
            true_hist = F.one_hot(targets, self.num_classes).float().sum(0, keepdim=True)
            E = torch.matmul(true_hist.T, pred_hist)
            E = E / (E.sum() + 1e-8)
            
            num = (self.w * O).sum()
            den = (self.w * E).sum()
            qwk_loss = 1 - (1 - num) / (1 - den + 1e-8)
            
            # Check for numerical issues
            if torch.isnan(qwk_loss) or torch.isinf(qwk_loss):
                print("Warning: NaN/Inf in QWK loss, using CORAL loss only")
                return coral_loss_value
                
            total_loss = self.alpha * coral_loss_value + (1 - self.alpha) * qwk_loss
            
        except Exception as e:
            print(f"Warning: Error in QWK loss computation: {e}, using CORAL loss only")
            total_loss = coral_loss_value

        return total_loss

# ==================== SAM OPTIMIZER (FIXED FOR AMP) ====================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer - AMP compatible"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute and store e_w"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: return to original point and update"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """Compute gradient norm"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

# ==================== TRAINING FUNCTIONS (PROPERLY FIXED) ====================
def train_epoch_sam(model, loader, criterion, optimizer, scaler, use_amp):
    """SAM training with proper AMP handling - disable AMP for SAM to avoid GradScaler issues"""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # Apply mixup/cutmix
        mixed = False
        if USE_MIXUP and USE_CUTMIX:
            if random.random() < MIXUP_PROB:
                if random.random() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                else:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                mixed = True
        elif USE_MIXUP:
            if random.random() < MIXUP_PROB:
                images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                mixed = True
        elif USE_CUTMIX:
            if random.random() < MIXUP_PROB:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                mixed = True

        # SAM WITHOUT AMP (to avoid GradScaler conflicts)
        # First forward-backward pass
        outputs = model(images)
        if mixed:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        outputs = model(images)
        if mixed:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.second_step(zero_grad=True)

        running_loss += loss.item()

    return running_loss / len(loader)

def train_epoch_standard(model, loader, criterion, optimizer, scaler, use_amp):
    """Standard training loop with AMP"""
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        mixed = False
        if USE_MIXUP and USE_CUTMIX:
            if random.random() < MIXUP_PROB:
                if random.random() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                else:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                mixed = True
        elif USE_MIXUP:
            if random.random() < MIXUP_PROB:
                images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                mixed = True
        elif USE_CUTMIX:
            if random.random() < MIXUP_PROB:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                mixed = True

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                if mixed:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if mixed:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, class_names, show_details=True):
    """Validation with CORAL or standard predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)

        if USE_CORAL:
            preds = coral_predict(outputs).cpu().numpy()
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = np.mean(all_labels == all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    if show_details:
        if SHOW_CONFUSION_MATRIX:
            print_confusion_matrix(cm, class_names)
        if SHOW_PER_CLASS_METRICS:
            print_per_class_metrics(all_labels, all_preds, class_names)

    return {'qwk': qwk, 'f1': f1, 'accuracy': acc, 'confusion_matrix': cm}, all_preds, all_labels

# ==================== UTILITY FUNCTIONS ====================
def print_confusion_matrix(cm, class_names):
    """Print formatted confusion matrix"""
    print("\n📊 Confusion Matrix:")
    header = "        " + "".join([f"{name:>10s}" for name in class_names])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>8s}"
        for val in row:
            row_str += f"{val:>10d}"
        print(row_str)
    print()

def print_per_class_metrics(y_true, y_pred, class_names):
    """Print per-class metrics"""
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

def load_data(data_path):
    """Load DR dataset"""
    folder_to_grade = {
        'No_DR': 0,
        'Mild': 1,
        'Moderate': 2,
        'Severe': 3,
        'Proliferate_DR': 4
    }

    all_paths = []
    all_labels = []
    samples_per_class = []

    print("\nLoading dataset...")
    for folder_name, grade in folder_to_grade.items():
        class_dir = data_path / folder_name
        if not class_dir.exists():
            print(f"⚠️ Warning: {class_dir} not found")
            samples_per_class.append(0)
            continue

        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            images.extend(list(class_dir.glob(ext)))

        all_paths.extend(images)
        all_labels.extend([grade] * len(images))
        samples_per_class.append(len(images))
        print(f"{folder_name} (Grade {grade}): {len(images)} images")

    print(f"\n✓ Total images: {len(all_paths)}")
    if len(all_paths) == 0:
        print(f"\n❌ ERROR: No images found in {data_path}")
        exit(1)

    return all_paths, all_labels, list(folder_to_grade.keys()), np.array(samples_per_class)

# ==================== MAIN ====================
def main():
    # Load data
    all_paths, all_labels, class_names, samples_per_class = load_data(DATA_PATH)

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
    print("PROPERLY FIXED TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {class_names}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"\nOptimizations:")
    print(f"  • SAM: {USE_SAM} (without AMP to avoid GradScaler conflicts)")
    print(f"  • SWA: {USE_SWA} (start epoch {SWA_START_EPOCH})")
    print(f"  • CORAL ordinal: {USE_CORAL} (numerically stable)")
    print(f"  • Class-balanced loss: {USE_CLASS_BALANCED}")
    print(f"  • Mixup: {USE_MIXUP} (α={MIXUP_ALPHA})")
    print(f"  • CutMix: {USE_CUTMIX} (α={CUTMIX_ALPHA})")

    # Initialize model
    print("\n✓ Using EfficientNet-B3 + CORAL (improved initialization)")
    model = EfficientNetDR_CORAL(num_classes=5).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Class-balanced weights with manual boosting
    if USE_CLASS_BALANCED:
        class_weights = get_class_weights(samples_per_class, CLASS_BALANCED_BETA).to(DEVICE)
        print(f"\nClass weights (before boost): {class_weights.cpu().numpy()}")

        # Manual boost for critical minority classes
        class_weights[3] *= SEVERE_WEIGHT_BOOST
        class_weights[4] *= PROLIFERATIVE_WEIGHT_BOOST

        # Renormalize
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        print(f"Class weights (after boost):  {class_weights.cpu().numpy()}")
        print(f"  • Severe (Grade 3): {class_weights[3].item():.3f} ({SEVERE_WEIGHT_BOOST}x boost)")
        print(f"  • Proliferative (Grade 4): {class_weights[4].item():.3f} ({PROLIFERATIVE_WEIGHT_BOOST}x boost)")
    else:
        class_weights = None

    # Loss function
    criterion = CombinedOrdinalLoss(num_classes=5, class_weights=class_weights, alpha=0.7)

    # Optimizer
    if USE_SAM:
        base_optimizer = lambda params, **kwargs: optim.AdamW(params, **kwargs)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=LR, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer if USE_SAM else optimizer,
        T_0=10, T_mult=2, eta_min=1e-6
    )

    # SWA setup
    if USE_SWA:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer.base_optimizer if USE_SAM else optimizer,
            swa_lr=SWA_LR
        )

    # Use AMP only for standard optimizer, not SAM
    use_amp = torch.cuda.is_available() and not USE_SAM
    scaler = GradScaler() if use_amp else None

    print(f"\nUsing AMP: {use_amp} (disabled for SAM to avoid conflicts)")

    best_qwk = 0
    patience = 20
    patience_counter = 0

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        # Train
        if USE_SAM:
            train_loss = train_epoch_sam(model, train_loader, criterion, optimizer, scaler, use_amp)
        else:
            train_loss = train_epoch_standard(model, train_loader, criterion, optimizer, scaler, use_amp)

        # Validate
        metrics, val_preds, val_labels_arr = validate_epoch(
            model, val_loader, class_names, show_details=True
        )

        # Step scheduler
        if USE_SWA and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        print(f"📊 EPOCH {epoch+1} SUMMARY")
        print(f"{'─'*60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val QWK: {metrics['qwk']:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        if USE_SWA and epoch >= SWA_START_EPOCH:
            print(f"SWA: Active (epoch {epoch - SWA_START_EPOCH + 1})")

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
                'confusion_matrix': metrics['confusion_matrix'],
                'config': {
                    'USE_SAM': USE_SAM,
                    'USE_SWA': USE_SWA,
                    'USE_CORAL': USE_CORAL,
                    'USE_CLASS_BALANCED': USE_CLASS_BALANCED,
                    'USE_MIXUP': USE_MIXUP,
                    'USE_CUTMIX': USE_CUTMIX,
                    'CLASS_BALANCED_BETA': CLASS_BALANCED_BETA,
                    'SEVERE_WEIGHT_BOOST': SEVERE_WEIGHT_BOOST,
                    'PROLIFERATIVE_WEIGHT_BOOST': PROLIFERATIVE_WEIGHT_BOOST
                }
            }, DATA_PATH.parent / 'best_dr_model_fixed.pth')
            print(f"✅ NEW BEST MODEL! QWK: {best_qwk:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience}")
            if patience_counter >= patience and epoch >= SWA_START_EPOCH:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    # Update BN statistics for SWA model
    if USE_SWA:
        print("\n" + "="*60)
        print("UPDATING SWA BATCH NORM STATISTICS")
        print("="*60)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

        # Save SWA model
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': swa_model.module.state_dict(),
            'best_qwk': best_qwk,
            'config': {
                'USE_SAM': USE_SAM,
                'USE_SWA': USE_SWA,
                'USE_CORAL': USE_CORAL,
                'USE_CLASS_BALANCED': USE_CLASS_BALANCED,
                'USE_MIXUP': USE_MIXUP,
                'USE_CUTMIX': USE_CUTMIX,
                'CLASS_BALANCED_BETA': CLASS_BALANCED_BETA,
                'SEVERE_WEIGHT_BOOST': SEVERE_WEIGHT_BOOST,
                'PROLIFERATIVE_WEIGHT_BOOST': PROLIFERATIVE_WEIGHT_BOOST
            }
        }, DATA_PATH.parent / 'swa_dr_model_fixed.pth')

        # Evaluate SWA model
        print("\nEvaluating SWA model...")
        swa_metrics, _, _ = validate_epoch(swa_model, val_loader, class_names, show_details=True)
        print(f"\n🎯 SWA MODEL VALIDATION")
        print(f"{'─'*60}")
        print(f"Val QWK: {swa_metrics['qwk']:.4f}")
        print(f"Val F1: {swa_metrics['f1']:.4f}")
        print(f"Val Accuracy: {swa_metrics['accuracy']*100:.2f}%")

        # Use SWA model for final test
        final_model = swa_model
    else:
        checkpoint = torch.load(DATA_PATH.parent / 'best_dr_model_fixed.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_model = model

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    test_metrics, test_preds, test_labels_arr = validate_epoch(
        final_model, test_loader, class_names, show_details=True
    )

    print(f"\n🎯 FINAL TEST RESULTS")
    print(f"{'─'*60}")
    print(f"Test QWK: {test_metrics['qwk']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Best Validation QWK: {best_qwk:.4f}")
    print(f"Final Test QWK: {test_metrics['qwk']:.4f}")
    print(f"Model saved to: {DATA_PATH.parent / ('swa_dr_model_fixed.pth' if USE_SWA else 'best_dr_model_fixed.pth')}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()