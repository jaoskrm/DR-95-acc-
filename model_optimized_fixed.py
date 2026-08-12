# ================================================================
# FIXED OPTIMIZED DR TRAINING - SAM + SWA + CORAL + XAI
# Fixed: SAM optimizer GradScaler workflow
# Added: Manual weight boosting for critical minority classes
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
CLASS_BALANCED_BETA = 0.99999  # Increased from 0.9999 for stronger minority focus

# MANUAL WEIGHT BOOSTS FOR CRITICAL CLASSES
SEVERE_WEIGHT_BOOST = 1.5      # Multiply Severe class weight by this
PROLIFERATIVE_WEIGHT_BOOST = 1.3  # Multiply Proliferative class weight by this

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
    """CORAL ordinal regression output layer"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.coral_weights = nn.Linear(512, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1).float())

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

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_head(features)

        if USE_CORAL:
            return self.ordinal_head(features)
        else:
            return self.classifier(features)

# ==================== XAI ====================
class XAIExplainer:
    """XAI explanations for DR model"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if hasattr(model, 'backbone'):
            if 'efficientnet' in str(type(model.backbone)).lower():
                self.target_layers = [model.backbone.blocks[-1][-1].conv_pwl]
            elif 'resnet' in str(type(model.backbone)).lower():
                self.target_layers = [model.backbone.layer4[-1]]

    def generate_gradcam(self, image_tensor, predicted_class=None, method='GradCAM'):
        cam_methods = {
            'GradCAM': GradCAM,
            'HiResCAM': HiResCAM,
            'ScoreCAM': ScoreCAM
        }

        cam = cam_methods.get(method, GradCAM)(
            model=self.model,
            target_layers=self.target_layers
        )

        if predicted_class is None:
            with torch.no_grad():
                outputs = self.model(image_tensor)
                if USE_CORAL:
                    predicted_class = coral_predict(outputs).item()
                else:
                    predicted_class = torch.argmax(outputs, dim=1).item()

        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

        rgb_img = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return cam_image, grayscale_cam

    def generate_lime_explanation(self, image_array, num_samples=1000):
        def batch_predict(images):
            self.model.eval()
            batch = []

            for img in images:
                transformed = test_transform(image=img)['image']
                batch.append(transformed)

            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                if USE_CORAL:
                    probs = coral_to_probs(outputs)
                else:
                    probs = torch.softmax(outputs, dim=1)

            return probs.cpu().numpy()

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image_array,
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples,
            batch_size=32
        )

        return explanation

    def visualize_lime(self, explanation, image_array, predicted_class):
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        img_boundry = mark_boundaries(temp / 255.0, mask)
        return img_boundry

# ==================== CORAL UTILITIES ====================
def coral_loss(logits, levels):
    """CORAL ordinal regression loss"""
    batch_size = logits.size(0)
    num_classes = logits.size(1) + 1

    levels_expanded = levels.unsqueeze(1).repeat(1, num_classes - 1)
    thresholds = torch.arange(num_classes - 1).unsqueeze(0).repeat(batch_size, 1).to(logits.device)

    targets = (levels_expanded > thresholds).float()

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    return loss

def coral_predict(logits):
    """Convert CORAL logits to class predictions"""
    probas = torch.sigmoid(logits)
    predictions = (probas > 0.5).sum(dim=1)
    return predictions

def coral_to_probs(logits):
    """Convert CORAL logits to class probabilities"""
    probas = torch.sigmoid(logits)
    probas = torch.cat([probas, torch.ones(probas.size(0), 1).to(probas.device)], dim=1)
    probas = torch.cat([torch.ones(probas.size(0), 1).to(probas.device), probas], dim=1)

    class_probs = probas[:, 1:] - probas[:, :-1]

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

# ==================== COMBINED LOSS ====================
class CombinedOrdinalLoss(nn.Module):
    """CORAL loss + QWK loss with class balancing"""
    def __init__(self, num_classes=5, class_weights=None, alpha=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_weights = class_weights

        w = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = (i - j)**2 / (num_classes - 1)**2
        self.w = torch.from_numpy(w).float().to(DEVICE)

    def forward(self, logits, targets):
        targets = targets.long()

        coral_loss_value = coral_loss(logits, targets)

        if self.class_weights is not None:
            weights_batch = self.class_weights[targets]
            coral_loss_value = (coral_loss_value * weights_batch.mean())

        probs = coral_to_probs(logits)
        O = torch.matmul(probs.T, F.one_hot(targets, self.num_classes).float())
        O = O / (O.sum() + 1e-8)
        pred_hist = probs.sum(0, keepdim=True)
        true_hist = F.one_hot(targets, self.num_classes).float().sum(0, keepdim=True)
        E = torch.matmul(true_hist.T, pred_hist)
        E = E / (E.sum() + 1e-8)
        num = (self.w * O).sum()
        den = (self.w * E).sum()
        qwk_loss = 1 - (1 - num) / (1 - den + 1e-8)

        return self.alpha * coral_loss_value + (1 - self.alpha) * qwk_loss

# ==================== SAM OPTIMIZER (FIXED) ====================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer"""
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

# ==================== TRAINING FUNCTIONS (FIXED) ====================
def train_epoch_sam(model, loader, criterion, optimizer, scaler, use_amp):
    """FIXED: Training loop with SAM optimizer and proper GradScaler handling"""
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

        # FIXED SAM WORKFLOW WITH AMP
        if use_amp:
            # First forward-backward pass
            with autocast():
                outputs = model(images)
                if mixed:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.first_step(zero_grad=True)
            scaler.update()  # Update scaler after first step

            # Second forward-backward pass (create new scaler state)
            with autocast():
                outputs = model(images)
                if mixed:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.second_step(zero_grad=True)
            scaler.update()  # Update scaler after second step
        else:
            # CPU fallback
            outputs = model(images)
            if mixed:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.first_step(zero_grad=True)

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
    """Standard training loop (no SAM)"""
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

# ==================== XAI VALIDATION ====================
@torch.no_grad()
def validate_with_xai(model, loader, class_names, save_dir, num_samples=5):
    """Enhanced validation with XAI visualizations"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    xai = XAIExplainer(model, DEVICE)

    print("\n" + "="*60)
    print("GENERATING XAI EXPLANATIONS")
    print("="*60)

    sample_count = 0

    for images, labels in loader:
        if sample_count >= num_samples:
            break

        for i in range(len(images)):
            if sample_count >= num_samples:
                break

            image_tensor = images[i:i+1].to(DEVICE)
            true_label = labels[i].item()

            outputs = model(image_tensor)
            if USE_CORAL:
                pred_class = coral_predict(outputs).item()
                probs = coral_to_probs(outputs)[0]
            else:
                pred_class = torch.argmax(outputs, dim=1).item()
                probs = torch.softmax(outputs, dim=1)[0]

            confidence = probs[pred_class].item()

            print(f"\n📊 Sample {sample_count + 1}/{num_samples}")
            print(f"True: {class_names[true_label]} | Pred: {class_names[pred_class]} ({confidence*100:.1f}%)")

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(
                f'XAI Analysis - True: {class_names[true_label]} | Pred: {class_names[pred_class]}',
                fontsize=16, fontweight='bold'
            )

            orig_img = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            orig_img = std * orig_img + mean
            orig_img = np.clip(orig_img, 0, 1)

            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')

            try:
                gradcam_img, _ = xai.generate_gradcam(image_tensor, pred_class, 'GradCAM')
                axes[0, 1].imshow(gradcam_img)
                axes[0, 1].set_title('Grad-CAM', fontweight='bold')
                axes[0, 1].axis('off')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center')
                axes[0, 1].axis('off')

            try:
                hirescam_img, _ = xai.generate_gradcam(image_tensor, pred_class, 'HiResCAM')
                axes[0, 2].imshow(hirescam_img)
                axes[0, 2].set_title('HiRes-CAM', fontweight='bold')
                axes[0, 2].axis('off')
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center')
                axes[0, 2].axis('off')

            try:
                lime_img = cv2.resize(orig_img, (224, 224))
                lime_img = (lime_img * 255).astype(np.uint8)
                explanation = xai.generate_lime_explanation(lime_img, num_samples=500)
                lime_vis = xai.visualize_lime(explanation, lime_img, pred_class)
                axes[1, 0].imshow(lime_vis)
                axes[1, 0].set_title('LIME', fontweight='bold')
                axes[1, 0].axis('off')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center')
                axes[1, 0].axis('off')

            axes[1, 1].barh(class_names, probs.cpu().numpy(), 
                           color=['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1'])
            axes[1, 1].set_xlabel('Probability', fontweight='bold')
            axes[1, 1].set_title('Class Probabilities', fontweight='bold')
            axes[1, 1].set_xlim(0, 1)

            for idx, (name, prob) in enumerate(zip(class_names, probs.cpu().numpy())):
                axes[1, 1].text(prob + 0.02, idx, f'{prob*100:.1f}%', va='center', fontsize=9)

            info_text = f"""
Model: {'EfficientNet-B3 + CORAL' if USE_CORAL else 'EfficientNet-B3'}
Optimizer: {'SAM' if USE_SAM else 'AdamW'}
Prediction: {class_names[pred_class]}
Confidence: {confidence*100:.2f}%
True Label: {class_names[true_label]}
Correct: {'✓' if pred_class == true_label else '✗'}

Top-3 Predictions:
"""
            top3_indices = torch.topk(probs, 3).indices
            for rank, idx in enumerate(top3_indices, 1):
                info_text += f"{rank}. {class_names[idx]}: {probs[idx]*100:.1f}%\n"

            axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, 
                           verticalalignment='center', family='monospace')
            axes[1, 2].axis('off')

            plt.tight_layout()

            save_path = save_dir / f'xai_sample_{sample_count+1}_{class_names[pred_class]}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved: {save_path}")

            sample_count += 1

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
    print("OPTIMIZED TRAINING CONFIGURATION (FIXED)")
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
    print(f"  • SAM: {USE_SAM} (FIXED GradScaler workflow)")
    print(f"  • SWA: {USE_SWA} (start epoch {SWA_START_EPOCH})")
    print(f"  • CORAL ordinal: {USE_CORAL}")
    print(f"  • Class-balanced loss: {USE_CLASS_BALANCED}")
    print(f"  • Mixup: {USE_MIXUP} (α={MIXUP_ALPHA})")
    print(f"  • CutMix: {USE_CUTMIX} (α={CUTMIX_ALPHA})")

    # Initialize model
    print("\n✓ Using EfficientNet-B3 + CORAL")
    model = EfficientNetDR_CORAL(num_classes=5).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Class-balanced weights with manual boosting
    if USE_CLASS_BALANCED:
        class_weights = get_class_weights(samples_per_class, CLASS_BALANCED_BETA).to(DEVICE)
        print(f"\nClass weights (before boost): {class_weights.cpu().numpy()}")

        # Manual boost for critical minority classes
        class_weights[3] *= SEVERE_WEIGHT_BOOST        # Severe (Grade 3)
        class_weights[4] *= PROLIFERATIVE_WEIGHT_BOOST  # Proliferative (Grade 4)

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

    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

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
            }, DATA_PATH.parent / 'best_dr_model_optimized.pth')
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
        }, DATA_PATH.parent / 'swa_dr_model_optimized.pth')

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
        checkpoint = torch.load(DATA_PATH.parent / 'best_dr_model_optimized.pth', weights_only=False)
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

    # Generate XAI visualizations
    print("\n" + "="*60)
    print("GENERATING XAI VISUALIZATIONS")
    print("="*60)

    xai_save_dir = DATA_PATH.parent / 'xai_visualizations_optimized'
    validate_with_xai(
        final_model,
        test_loader,
        class_names,
        xai_save_dir,
        num_samples=10
    )

    print(f"\n✅ XAI visualizations saved to: {xai_save_dir}")
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Best Validation QWK: {best_qwk:.4f}")
    print(f"Final Test QWK: {test_metrics['qwk']:.4f}")
    print(f"Model saved to: {DATA_PATH.parent / ('swa_dr_model_optimized.pth' if USE_SWA else 'best_dr_model_optimized.pth')}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()