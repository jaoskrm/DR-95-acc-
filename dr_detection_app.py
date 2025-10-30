# dr_detection_app_optimized_xai.py
# DR Detection with Working XAI Methods

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

# Core imports
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import timm
import io

# XAI Libraries
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import Occlusion

# LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic


def clear_gpu_memory():
    """Aggressively clear GPU memory with synchronization"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except:
            pass


# Page configuration
st.set_page_config(
    page_title="DR Detection with Advanced XAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stAlert {
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==================== MODEL ARCHITECTURE ====================
class EfficientNetDR_V2(nn.Module):
    """EfficientNet-B5 for DR Detection"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b5', pretrained=False)
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== PREPROCESSING FUNCTIONS ====================
def crop_image_from_gray(img, tol=7):
    """Remove black borders from fundus images"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img, sigmaX=10, img_size=384):
    """Ben Graham's circle cropping"""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return Image.fromarray(img)


def enhance_fundus_image(image):
    """CLAHE enhancement"""
    try:
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    except:
        return image


def preprocess_image(image):
    """Full preprocessing pipeline"""
    cropped = circle_crop(image)
    enhanced = enhance_fundus_image(cropped)
    
    img_array = np.array(enhanced).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    
    return img_tensor, cropped, enhanced


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_ensemble_models(model_dir, load_all=True):
    """Load models - option to load only best for XAI"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    
    num_folds = 5 if load_all else 1
    
    for fold in range(num_folds):
        model_path = Path(model_dir) / f'best_model_fold{fold}.pth'
        
        if not model_path.exists():
            st.error(f"❌ Model file not found: {model_path}")
            return None, device
        
        checkpoint = torch.load(model_path, map_location=device)
        model = EfficientNetDR_V2(num_classes=5).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
        del checkpoint
        torch.cuda.empty_cache()
    
    return models, device


# ==================== XAI 1: GRAD-CAM++ ====================
def generate_gradcam_plusplus(model, img_tensor, predicted_class, device):
    """Generate Grad-CAM++ visualization"""
    try:
        target_layers = [model.backbone.blocks[-1][-1].conv_pwl]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class)]
        
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device), 
                           targets=targets)[0]
        return grayscale_cam
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM++ failed: {str(e)}")
        return None


# ==================== XAI 2: GRADCAM (REGULAR) ====================
def generate_gradcam(model, img_tensor, predicted_class, device):
    """Regular GradCAM - complementary to Grad-CAM++"""
    try:
        target_layers = [model.backbone.blocks[-1][-1].conv_pwl]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class)]
        
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device), 
                           targets=targets)[0]
        return grayscale_cam
    except Exception as e:
        st.warning(f"⚠️ GradCAM failed: {str(e)}")
        return None


# ==================== XAI 3: LAYER-CAM ====================
def generate_layercam(model, img_tensor, predicted_class, device):
    """LayerCAM - better localization than GradCAM"""
    try:
        from pytorch_grad_cam import LayerCAM
        
        target_layers = [model.backbone.blocks[-1][-1].conv_pwl]
        cam = LayerCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class)]
        
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device), 
                           targets=targets)[0]
        return grayscale_cam
    except Exception as e:
        st.warning(f"⚠️ LayerCAM failed: {str(e)}")
        return None


# ==================== XAI 4: OCCLUSION SENSITIVITY ====================
def generate_occlusion_map(model, img_tensor, predicted_class, device):
    """Occlusion-based attribution"""
    try:
        model.eval()
        clear_gpu_memory()
        
        occlusion = Occlusion(model)
        img_input = img_tensor.unsqueeze(0).to(device)
        
        attribution = occlusion.attribute(
            img_input,
            target=predicted_class,
            strides=(3, 8, 8),
            sliding_window_shapes=(3, 24, 24),
            baselines=0
        )
        
        attribution = attribution.squeeze().cpu().detach().numpy()
        attribution = np.abs(attribution).sum(axis=0)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        clear_gpu_memory()
        return attribution
    
    except Exception as e:
        st.warning(f"⚠️ Occlusion failed: {str(e)}")
        clear_gpu_memory()
        return None


# ==================== XAI 5: LIME ====================
def generate_lime_explanation(model, img_array, predicted_class, device, num_samples=1000):
    """Generate LIME explanation with superpixel segmentation"""
    try:
        model.eval()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = img_array.transpose(1, 2, 0) * std + mean
        img_display = np.clip(img_display, 0, 1)
        
        def predict_fn(images):
            batch = []
            for img in images:
                img_norm = (img - mean) / std
                img_norm = img_norm.transpose(2, 0, 1)
                batch.append(img_norm)
            
            batch_tensor = torch.FloatTensor(np.array(batch)).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            return probs
        
        explainer = lime_image.LimeImageExplainer()
        
        explanation = explainer.explain_instance(
            img_display,
            predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=lambda x: slic(x, n_segments=100, compactness=10, sigma=1)
        )
        
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        return mask, img_display
    
    except Exception as e:
        st.warning(f"⚠️ LIME generation failed: {str(e)}")
        return None, None


# ==================== XAI 6: ADVERSARIAL EXAMPLES ====================
def generate_adversarial_fgsm(model, img_tensor, true_label, device, epsilon=0.03):
    """Generate FGSM adversarial example"""
    try:
        model.eval()
        
        img_input = img_tensor.unsqueeze(0).clone().detach().to(device).requires_grad_(True)
        
        output = model(img_input)
        loss = F.cross_entropy(output, torch.tensor([true_label]).to(device))
        
        model.zero_grad()
        loss.backward()
        
        data_grad = img_input.grad.data.clone()
        sign_data_grad = data_grad.sign()
        perturbed_image = (img_input.detach() + epsilon * sign_data_grad).clamp(img_input.min(), img_input.max())
        
        with torch.no_grad():
            orig_pred = torch.softmax(output.detach(), dim=1)
            adv_output = model(perturbed_image)
            adv_pred = torch.softmax(adv_output, dim=1)
        
        return perturbed_image.squeeze(0).detach().cpu(), orig_pred[0].cpu().numpy(), adv_pred[0].cpu().numpy()
    
    except Exception as e:
        st.warning(f"⚠️ Adversarial generation failed: {str(e)}")
        return None, None, None


# ==================== VISUALIZATION HELPERS ====================
def create_heatmap_overlay(original_img, cam, alpha=0.5):
    """Create colored heatmap overlay"""
    img_np = np.array(original_img).astype(np.float32) / 255.0
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    overlayed = heatmap * alpha + img_np * (1 - alpha)
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


def denormalize_tensor(tensor):
    """Convert normalized tensor to displayable image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    return img


# ==================== PREDICTION ====================
def predict_with_ensemble(models, image_tensor, device):
    """Ensemble prediction"""
    image_tensor_input = image_tensor.unsqueeze(0).to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for model in models:
            output = model(image_tensor_input)
            probs = torch.softmax(output, dim=1)
            all_predictions.append(probs)
    
    ensemble_probs = torch.stack(all_predictions).mean(dim=0).squeeze()
    predicted_class = torch.argmax(ensemble_probs).item()
    confidence = ensemble_probs[predicted_class].item()
    
    return predicted_class, confidence, ensemble_probs.cpu().numpy()


# ==================== PROBABILITY CHART ====================
def create_probability_chart(probabilities, class_names, predicted_class):
    """Create probability bar chart"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
    bars = ax.barh(class_names, probabilities, color=colors, alpha=0.7, edgecolor='black')
    
    bars[predicted_class].set_color('gold')
    bars[predicted_class].set_edgecolor('red')
    bars[predicted_class].set_linewidth(3)
    
    ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
    ax.set_title('Ensemble Prediction Probabilities', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    for idx, prob in enumerate(probabilities):
        ax.text(prob + 0.02, idx, f'{prob*100:.1f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==================== ADVERSARIAL COMPARISON ====================
def create_adversarial_comparison(original_img, adv_img, orig_probs, adv_probs, class_names):
    """Create adversarial robustness comparison"""
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(denormalize_tensor(original_img))
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(denormalize_tensor(adv_img))
    ax2.set_title('Adversarial Image (FGSM ε=0.03)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, orig_probs, width, label='Original', alpha=0.8, color='steelblue')
    bars2 = ax3.bar(x + width/2, adv_probs, width, label='Adversarial', alpha=0.8, color='coral')
    
    ax3.set_ylabel('Probability', fontweight='bold')
    ax3.set_title('Prediction Shift', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==================== SEVERITY BADGE ====================
def get_severity_badge(predicted_class, class_names):
    """Get colored severity badge"""
    severity_colors = {
        0: ("🟢", "background-color: #d4edda; color: #155724; padding: 10px 20px; border-radius: 10px; font-size: 18px;"),
        1: ("🟡", "background-color: #fff3cd; color: #856404; padding: 10px 20px; border-radius: 10px; font-size: 18px;"),
        2: ("🟠", "background-color: #ffeaa7; color: #d63031; padding: 10px 20px; border-radius: 10px; font-size: 18px;"),
        3: ("🔴", "background-color: #f8d7da; color: #721c24; padding: 10px 20px; border-radius: 10px; font-size: 18px;"),
        4: ("🔴", "background-color: #d63031; color: white; padding: 10px 20px; border-radius: 10px; font-size: 18px;")
    }
    
    icon, style = severity_colors[predicted_class]
    return f"<div style='{style}'><b>{icon} {class_names[predicted_class]}</b></div>"


# ==================== MAIN APP ====================
def main():
    st.title("👁️ Diabetic Retinopathy Detection with Advanced XAI")
    st.markdown("### 🔬 Grad-CAM++ • GradCAM • LayerCAM • LIME • Occlusion • Adversarial")
    
    st.sidebar.header("⚙️ Configuration")
    
    model_dir = st.sidebar.text_input(
        "Model Directory",
        value="D:/studies/clg/cao/data",
        help="Path to folder containing model checkpoint files"
    )
    
    load_all_models = st.sidebar.checkbox("Load All 5 Models", value=False, 
                                          help="Use only for ensemble prediction. Disable for XAI to save memory.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("XAI Methods")
    
    enable_gradcam_pp = st.sidebar.checkbox("🔥 Grad-CAM++", value=True, 
                                            help="Best overall activation mapping")
    enable_gradcam = st.sidebar.checkbox("🔥 GradCAM", value=False, 
                                         help="Complementary to Grad-CAM++")
    enable_layercam = st.sidebar.checkbox("🔥 LayerCAM", value=False, 
                                          help="Better localization")
    enable_lime = st.sidebar.checkbox("🍋 LIME", value=True, 
                                      help="Model-agnostic explanations")
    enable_occlusion = st.sidebar.checkbox("🔲 Occlusion", value=False, 
                                           help="Slow but interpretable (2-3 min)")
    enable_adversarial = st.sidebar.checkbox("⚡ Adversarial Testing", value=True, 
                                             help="FGSM robustness check")
    
    if enable_lime:
        lime_samples = st.sidebar.slider("LIME samples", 500, 2000, 1000, 
                                         help="More samples = better but slower")
    else:
        lime_samples = 1000
    
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    
    st.sidebar.markdown("---")
    with st.spinner("🔄 Loading ensemble models..."):
        models, device = load_ensemble_models(model_dir, load_all=load_all_models)
    
    if models is None:
        st.error("❌ Failed to load models! Check the model directory path.")
        return
    
    st.sidebar.success(f"✅ Loaded {len(models)} model(s) on **{device}**")
    
    st.markdown("---")
    st.header("📤 Upload Retinal Fundus Image")
    
    uploaded_file = st.file_uploader(
        "Choose a fundus image for analysis", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a color fundus photograph"
    )
    
    if uploaded_file is not None:
        process_image_with_xai(
            uploaded_file, models, device, class_names,
            enable_gradcam_pp, enable_gradcam, enable_layercam,
            enable_lime, enable_occlusion, enable_adversarial, lime_samples
        )


def process_image_with_xai(uploaded_file, models, device, class_names,
                          enable_gradcam_pp, enable_gradcam, enable_layercam,
                          enable_lime, enable_occlusion, enable_adversarial, lime_samples):
    """Memory-optimized XAI processing"""
    
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    
    with st.spinner("⚙️ Preprocessing image..."):
        img_tensor, cropped, enhanced = preprocess_image(image)
    
    with col2:
        st.subheader("✨ Enhanced Image")
        st.image(enhanced, use_container_width=True)
    
    with st.spinner("🤖 Running ensemble prediction..."):
        predicted_class, confidence, probabilities = predict_with_ensemble(models, img_tensor, device)
    
    st.markdown("---")
    st.header("🎯 Diagnosis Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Predicted Class")
        st.markdown(get_severity_badge(predicted_class, class_names), unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence Score", f"{confidence*100:.1f}%", 
                 delta=f"{(confidence - 0.5)*100:.1f}% vs random" if confidence > 0.5 else None)
    
    with col3:
        risk = "Low" if predicted_class <= 1 else ("High" if predicted_class >= 3 else "Moderate")
        risk_color = "green" if predicted_class <= 1 else ("red" if predicted_class >= 3 else "orange")
        st.markdown(f"### Risk Level")
        st.markdown(f"<h2 style='color: {risk_color};'>{risk}</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Class Probability Distribution")
    fig_prob = create_probability_chart(probabilities, class_names, predicted_class)
    st.pyplot(fig_prob)
    plt.close(fig_prob)
    
    st.markdown("---")
    st.header("🔍 Explainable AI (XAI) Analysis")
    
    best_model = models[0]
    
    if len(models) > 1:
        for i in range(len(models) - 1, 0, -1):
            try:
                models[i].cpu()
                del models[i]
            except:
                pass
        models = [best_model]
    
    clear_gpu_memory()
    
    # ========== GRAD-CAM++ ==========
    if enable_gradcam_pp:
        with st.spinner("Generating Grad-CAM++..."):
            gradcam_pp = generate_gradcam_plusplus(best_model, img_tensor, predicted_class, device)
            
            if gradcam_pp is not None:
                fig_gc = plt.figure(figsize=(12, 4))
                
                ax1 = fig_gc.add_subplot(1, 3, 1)
                ax1.imshow(enhanced)
                ax1.set_title('Enhanced Image')
                ax1.axis('off')
                
                ax2 = fig_gc.add_subplot(1, 3, 2)
                gradcam_overlay = create_heatmap_overlay(enhanced, gradcam_pp, alpha=0.5)
                ax2.imshow(gradcam_overlay)
                ax2.set_title('Grad-CAM++ Activation')
                ax2.axis('off')
                
                ax3 = fig_gc.add_subplot(1, 3, 3)
                im = ax3.imshow(gradcam_pp, cmap='jet')
                ax3.set_title('Raw Heatmap')
                ax3.axis('off')
                plt.colorbar(im, ax=ax3, fraction=0.046)
                
                st.pyplot(fig_gc)
                plt.close(fig_gc)
            clear_gpu_memory()
    
    # ========== GRADCAM ==========
    if enable_gradcam:
        with st.spinner("Generating GradCAM..."):
            gradcam = generate_gradcam(best_model, img_tensor, predicted_class, device)
            
            if gradcam is not None:
                fig_gc = plt.figure(figsize=(12, 4))
                
                ax1 = fig_gc.add_subplot(1, 2, 1)
                gradcam_overlay = create_heatmap_overlay(enhanced, gradcam, alpha=0.5)
                ax1.imshow(gradcam_overlay)
                ax1.set_title('GradCAM Overlay')
                ax1.axis('off')
                
                ax2 = fig_gc.add_subplot(1, 2, 2)
                im = ax2.imshow(gradcam, cmap='jet')
                ax2.set_title('Raw GradCAM')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046)
                
                st.pyplot(fig_gc)
                plt.close(fig_gc)
            clear_gpu_memory()
    
    # ========== LAYERCAM ==========
    if enable_layercam:
        with st.spinner("Generating LayerCAM..."):
            layercam = generate_layercam(best_model, img_tensor, predicted_class, device)
            
            if layercam is not None:
                fig_lc = plt.figure(figsize=(12, 4))
                
                ax1 = fig_lc.add_subplot(1, 2, 1)
                layercam_overlay = create_heatmap_overlay(enhanced, layercam, alpha=0.5)
                ax1.imshow(layercam_overlay)
                ax1.set_title('LayerCAM Overlay')
                ax1.axis('off')
                
                ax2 = fig_lc.add_subplot(1, 2, 2)
                im = ax2.imshow(layercam, cmap='jet')
                ax2.set_title('Raw LayerCAM')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046)
                
                st.pyplot(fig_lc)
                plt.close(fig_lc)
            clear_gpu_memory()
    
    # ========== LIME ==========
    if enable_lime:
        with st.spinner(f"Generating LIME (this may take 1-2 minutes)..."):
            try:
                lime_samples = min(lime_samples, 500)
                img_array = img_tensor.cpu().numpy()
                lime_mask, lime_img = generate_lime_explanation(
                    best_model, img_array, predicted_class, device, num_samples=lime_samples
                )
                
                if lime_mask is not None and lime_img is not None:
                    fig_lime = plt.figure(figsize=(12, 4))
                    
                    ax1 = fig_lime.add_subplot(1, 2, 1)
                    lime_overlay = mark_boundaries(lime_img, lime_mask, color=(0, 1, 0), mode='thick')
                    ax1.imshow(lime_overlay)
                    ax1.set_title('LIME Superpixel Explanation')
                    ax1.axis('off')
                    
                    ax2 = fig_lime.add_subplot(1, 2, 2)
                    ax2.imshow(lime_mask, cmap='gray')
                    ax2.set_title('Important Regions (White)')
                    ax2.axis('off')
                    
                    st.pyplot(fig_lime)
                    plt.close(fig_lime)
                clear_gpu_memory()
            except Exception as e:
                st.error(f"LIME failed: {str(e)}")
                clear_gpu_memory()
    
    # ========== OCCLUSION ==========
    if enable_occlusion:
        with st.spinner("Generating Occlusion Map (this may take 2-3 minutes)..."):
            occlusion_map = generate_occlusion_map(best_model, img_tensor, predicted_class, device)
            
            if occlusion_map is not None:
                fig_occ = plt.figure(figsize=(12, 4))
                
                ax1 = fig_occ.add_subplot(1, 2, 1)
                occlusion_overlay = create_heatmap_overlay(enhanced, occlusion_map, alpha=0.5)
                ax1.imshow(occlusion_overlay)
                ax1.set_title('Occlusion Sensitivity Overlay')
                ax1.axis('off')
                
                ax2 = fig_occ.add_subplot(1, 2, 2)
                im = ax2.imshow(occlusion_map, cmap='jet')
                ax2.set_title('Raw Occlusion Map')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046)
                
                st.pyplot(fig_occ)
                plt.close(fig_occ)
            clear_gpu_memory()
    
    # ========== ADVERSARIAL ==========
    if enable_adversarial:
        clear_gpu_memory()
        with st.spinner("Generating adversarial example..."):
            adv_tensor, orig_probs, adv_probs = generate_adversarial_fgsm(
                best_model, img_tensor, predicted_class, device, epsilon=0.03
            )
            
            if adv_tensor is not None:
                fig_adv = create_adversarial_comparison(
                    img_tensor, adv_tensor, orig_probs, adv_probs, class_names
                )
                st.pyplot(fig_adv)
                plt.close(fig_adv)
                clear_gpu_memory()
    
    # Explanation guide
    with st.expander("📖 How to Interpret XAI Visualizations"):
        st.markdown("""
        ### XAI Method Comparison
        
        **🔥 Grad-CAM++** - Best overall, shows discriminative regions with weighted gradients
        
        **🔥 GradCAM** - Standard CAM, complements Grad-CAM++ 
        
        **🔥 LayerCAM** - Better spatial localization than GradCAM
        
        **🍋 LIME** - Model-agnostic, superpixel-based explanations
        
        **🔲 Occlusion** - Masks regions to test importance (slow but accurate)
        
        **⚡ Adversarial** - Tests model robustness to small perturbations
        
        ### Why No Gradient-Based Methods?
        Saliency, SmoothGrad, and DeepLIFT **don't work** with your EfficientNet-B5 
        classifier (Dropout + BatchNorm kills gradients). CAM-based methods work 
        because they use convolutional layer activations, not classifier gradients.
        
        ### Performance Tips
        - Grad-CAM++ / GradCAM / LayerCAM: < 1 sec
        - LIME: 1-2 minutes
        - Occlusion: 2-3 minutes
        """)
    
    # Clinical recommendations
    st.markdown("---")
    st.header("💊 Clinical Recommendations")
    
    recommendations = {
        0: {
            "text": "✅ **No diabetic retinopathy detected**",
            "action": "Continue routine annual diabetic eye screening",
            "urgency": "low"
        },
        1: {
            "text": "⚠️ **Mild non-proliferative DR detected**",
            "action": "Schedule follow-up examination in 6-12 months",
            "urgency": "low"
        },
        2: {
            "text": "⚠️ **Moderate non-proliferative DR detected**",
            "action": "Refer to ophthalmologist within 3-6 months",
            "urgency": "moderate"
        },
        3: {
            "text": "🚨 **Severe non-proliferative DR detected**",
            "action": "URGENT referral to retinal specialist within 1-2 weeks",
            "urgency": "high"
        },
        4: {
            "text": "🚨 **Proliferative DR detected**",
            "action": "IMMEDIATE referral for laser/anti-VEGF therapy",
            "urgency": "critical"
        }
    }
    
    rec = recommendations[predicted_class]
    
    if rec["urgency"] in ["high", "critical"]:
        st.error(f"**{rec['text']}**\n\n{rec['action']}")
    elif rec["urgency"] == "moderate":
        st.warning(f"**{rec['text']}**\n\n{rec['action']}")
    else:
        st.success(f"**{rec['text']}**\n\n{rec['action']}")


if __name__ == "__main__":
    main()