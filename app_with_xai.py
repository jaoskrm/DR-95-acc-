# ================================================================
# DIABETIC RETINOPATHY WEB APP - STREAMLIT WITH XAI
# Single + Batch Image Testing with Explainable AI
# ================================================================

import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import zipfile

# XAI IMPORTS
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="DR Classifier with XAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODEL ARCHITECTURE ====================
class EfficientNetDR(nn.Module):
    """EfficientNet-B3 for DR classification"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False)
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

# ==================== PREPROCESSING ====================
def enhance_fundus_image(image):
    """CLAHE preprocessing for fundus images"""
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

# Transform
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model(model_path, device='cuda'):
    """Load trained model (cached)"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = EfficientNetDR(num_classes=5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ==================== PREDICTION ====================
def predict_image(model, image, device='cuda', apply_preprocessing=True):
    """Predict DR severity for a single image"""
    # Preprocess
    if apply_preprocessing:
        image = enhance_fundus_image(image)

    # Transform
    image_array = np.array(image)
    transformed = test_transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(outputs, dim=1).item()

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    return {
        'class': predicted_class,
        'label': class_names[predicted_class],
        'probabilities': probabilities.cpu().numpy(),
        'confidence': probabilities[predicted_class].item(),
        'image_tensor': image_tensor,
        'original_array': image_array
    }

# ==================== XAI FUNCTIONS ====================
def get_gradcam_visualization(model, image_tensor, predicted_class, method='GradCAM', device='cuda'):
    """Generate Grad-CAM or HiRes-CAM visualization"""
    # Get target layer
    if hasattr(model, 'backbone'):
        if 'efficientnet' in str(type(model.backbone)).lower():
            target_layers = [model.backbone.blocks[-1][-1].conv_pwl]
        else:
            target_layers = [model.backbone.layer4[-1]]

    # Select CAM method
    cam_methods = {'GradCAM': GradCAM, 'HiResCAM': HiResCAM, 'ScoreCAM': ScoreCAM}
    cam = cam_methods.get(method, GradCAM)(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    # Denormalize image
    rgb_img = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image, grayscale_cam

def get_lime_explanation(model, image_array, predicted_class, device='cuda'):
    """Generate LIME explanation"""
    def batch_predict(images):
        model.eval()
        batch = []

        for img in images:
            transformed = test_transform(image=img)['image']
            batch.append(transformed)

        batch_tensor = torch.stack(batch).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array,
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=500,
        batch_size=32
    )

    # Get visualization
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    lime_vis = mark_boundaries(temp / 255.0, mask)
    return lime_vis

# ==================== VISUALIZATION ====================
def create_probability_chart(probabilities):
    """Create interactive bar chart of probabilities"""
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']

    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Severity Level",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )

    return fig

def get_severity_color(label):
    """Get color based on severity"""
    colors = {
        'No DR': '#28a745',
        'Mild': '#ffc107',
        'Moderate': '#fd7e14',
        'Severe': '#dc3545',
        'Proliferative DR': '#6f42c1'
    }
    return colors.get(label, '#6c757d')

def get_recommendation(label):
    """Get clinical recommendation"""
    recommendations = {
        'No DR': '✅ No diabetic retinopathy detected. Continue regular eye exams annually.',
        'Mild': '⚠️ Mild DR detected. Recommend follow-up in 6-12 months.',
        'Moderate': '⚠️ Moderate DR detected. Recommend ophthalmologist consultation within 3-6 months.',
        'Severe': '🚨 Severe DR detected. Urgent ophthalmologist consultation recommended within 1 month.',
        'Proliferative DR': '🚨 Proliferative DR detected. Immediate ophthalmologist consultation required.'
    }
    return recommendations.get(label, 'Consult healthcare professional.')

# ==================== BATCH PROCESSING ====================
def process_batch(model, images, device='cuda', apply_preprocessing=True):
    """Process multiple images"""
    results = []
    progress_bar = st.progress(0)

    for idx, (name, img) in enumerate(images):
        try:
            result = predict_image(model, img, device, apply_preprocessing)
            results.append({
                'filename': name,
                'prediction': result['label'],
                'confidence': result['confidence'],
                'no_dr': result['probabilities'][0],
                'mild': result['probabilities'][1],
                'moderate': result['probabilities'][2],
                'severe': result['probabilities'][3],
                'proliferative': result['probabilities'][4]
            })
        except Exception as e:
            results.append({
                'filename': name,
                'prediction': 'Error',
                'confidence': 0,
                'no_dr': 0,
                'mild': 0,
                'moderate': 0,
                'severe': 0,
                'proliferative': 0
            })
        progress_bar.progress((idx + 1) / len(images))

    return pd.DataFrame(results)

# ==================== MAIN APP ====================
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Model path
    MODEL_PATH = st.sidebar.text_input(
        "Model Path",
        value=r"D:\studies\clg\cao\data\best_dr_model.pth"
    )

    # Device selection
    device_options = ['cuda', 'cpu']
    device = st.sidebar.selectbox(
        "Device",
        device_options,
        index=0 if torch.cuda.is_available() else 1
    )

    # Preprocessing toggle
    apply_preprocessing = st.sidebar.checkbox("Apply CLAHE Preprocessing", value=True)

    # XAI toggle
    show_xai = st.sidebar.checkbox("🔍 Show XAI Visualizations", value=True)

    # Load model
    if Path(MODEL_PATH).exists():
        model, checkpoint = load_model(MODEL_PATH, device)
        if model is not None:
            st.sidebar.success("✅ Model Loaded!")
            st.sidebar.metric("Best QWK", f"{checkpoint['best_qwk']:.4f}")
            st.sidebar.metric("Trained Epoch", checkpoint['epoch'] + 1)
    else:
        st.sidebar.error(f"❌ Model not found at:\n{MODEL_PATH}")
        model = None

    # Main content
    st.title("👁️ Diabetic Retinopathy Classifier with XAI")
    st.markdown("### State-of-the-Art DR Detection with Explainable AI")

    # Mode selection
    tab1, tab2, tab3 = st.tabs(["🖼️ Single Image", "📁 Batch Processing", "ℹ️ About"])

    # ==================== TAB 1: SINGLE IMAGE WITH XAI ====================
    with tab1:
        st.header("Single Image Prediction with Explainable AI")

        uploaded_file = st.file_uploader(
            "Upload a fundus image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a retinal fundus photograph"
        )

        if uploaded_file is not None and model is not None:
            # Display original image
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Preprocessed Image")
                if apply_preprocessing:
                    enhanced = enhance_fundus_image(image)
                    st.image(enhanced, use_column_width=True)
                else:
                    st.info("Preprocessing disabled")

            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = predict_image(model, image, device, apply_preprocessing)

                # Results
                st.markdown("---")

                # Main prediction
                color = get_severity_color(result['label'])
                st.markdown(
                    f"<h2 style='text-align: center; color: {color};'>"
                    f"Prediction: {result['label']}</h2>",
                    unsafe_allow_html=True
                )

                # Confidence
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%",
                    delta=None
                )

                # Recommendation
                st.info(get_recommendation(result['label']))

                # Probability chart
                fig = create_probability_chart(result['probabilities'])
                st.plotly_chart(fig, use_container_width=True)

                # ==================== XAI VISUALIZATIONS ====================
                if show_xai:
                    st.markdown("---")
                    st.subheader("🔬 Explainable AI Visualizations")
                    st.caption(
                        "These visualizations show which regions of the retinal image "
                        "influenced the model's prediction, providing transparency and trust."
                    )

                    with st.spinner("Generating XAI explanations..."):
                        # Create tabs for different XAI methods
                        xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs(
                            ["Grad-CAM", "HiRes-CAM", "Score-CAM", "LIME"]
                        )

                        with xai_tab1:
                            try:
                                st.markdown("**Gradient-weighted Class Activation Mapping**")
                                st.caption(
                                    "Highlights the regions of the image that most influenced "
                                    "the model's prediction using gradient information. Red areas "
                                    "indicate high importance, blue areas indicate low importance."
                                )
                                gradcam_img, _ = get_gradcam_visualization(
                                    model, result['image_tensor'], result['class'], 'GradCAM', device
                                )
                                st.image(gradcam_img, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error generating Grad-CAM: {str(e)}")

                        with xai_tab2:
                            try:
                                st.markdown("**High-Resolution Class Activation Mapping**")
                                st.caption(
                                    "Provides higher resolution attention maps showing "
                                    "fine-grained details of model focus areas. Useful for "
                                    "identifying small lesions and microaneurysms."
                                )
                                hirescam_img, _ = get_gradcam_visualization(
                                    model, result['image_tensor'], result['class'], 'HiResCAM', device
                                )
                                st.image(hirescam_img, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error generating HiRes-CAM: {str(e)}")

                        with xai_tab3:
                            try:
                                st.markdown("**Score-weighted Class Activation Mapping**")
                                st.caption(
                                    "Uses forward pass scores to weight feature maps, "
                                    "providing gradient-free visualization. Useful for "
                                    "comparing with gradient-based methods."
                                )
                                scorecam_img, _ = get_gradcam_visualization(
                                    model, result['image_tensor'], result['class'], 'ScoreCAM', device
                                )
                                st.image(scorecam_img, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error generating Score-CAM: {str(e)}")

                        with xai_tab4:
                            try:
                                st.markdown("**Local Interpretable Model-agnostic Explanations**")
                                st.caption(
                                    "Shows which superpixels (image regions) contributed "
                                    "most to the prediction through perturbation analysis. "
                                    "Green boundaries indicate important regions."
                                )

                                lime_img = cv2.resize(result['original_array'], (224, 224))
                                lime_vis = get_lime_explanation(
                                    model, lime_img, result['class'], device
                                )
                                st.image(lime_vis, use_column_width=True)

                            except Exception as e:
                                st.error(f"Error generating LIME: {str(e)}")

                    # Clinical interpretation guide
                    with st.expander("📖 How to Interpret XAI Visualizations"):
                        st.markdown("""
                        ### Understanding the Visualizations

                        **Grad-CAM, HiRes-CAM, Score-CAM:**
                        - **Red/Yellow regions**: Areas that strongly influenced the prediction
                        - **Blue/Green regions**: Areas with moderate influence
                        - **Dark regions**: Areas with minimal influence
                        - Look for focus on:
                          - Blood vessels and their abnormalities
                          - Hemorrhages (red spots)
                          - Microaneurysms (small red dots)
                          - Exudates (yellow/white patches)
                          - Cotton-wool spots

                        **LIME:**
                        - **Green boundaries**: Superpixels that contributed positively to the prediction
                        - Multiple highlighted regions indicate distributed features
                        - Absence of highlights in expected areas may indicate model uncertainty

                        ### Clinical Validation
                        - Compare XAI highlights with known DR lesion locations
                        - Verify model focuses on clinically relevant features
                        - Use as supplementary evidence, not sole diagnostic criterion
                        - Consult ophthalmologist for final diagnosis
                        """)

    # ==================== TAB 2: BATCH PROCESSING ====================
    with tab2:
        st.header("Batch Image Processing")

        uploaded_files = st.file_uploader(
            "Upload multiple fundus images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple retinal fundus photographs for batch analysis"
        )

        if uploaded_files and model is not None:
            st.write(f"📊 {len(uploaded_files)} images uploaded")

            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    # Load images
                    images = [(f.name, Image.open(f).convert('RGB')) for f in uploaded_files]

                    # Process
                    results_df = process_batch(model, images, device, apply_preprocessing)

                # Display results
                st.success(f"✅ Processed {len(results_df)} images")

                # Summary statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    no_dr_count = len(results_df[results_df['prediction'] == 'No DR'])
                    st.metric("No DR", no_dr_count)

                with col2:
                    mild_mod = len(results_df[results_df['prediction'].isin(['Mild', 'Moderate'])])
                    st.metric("Mild/Moderate", mild_mod)

                with col3:
                    severe = len(results_df[results_df['prediction'].isin(['Severe', 'Proliferative DR'])])
                    st.metric("Severe/Proliferative", severe)

                # Distribution chart
                pred_counts = results_df['prediction'].value_counts()
                fig = go.Figure(data=[
                    go.Bar(
                        x=pred_counts.index,
                        y=pred_counts.values,
                        marker_color=['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1'][:len(pred_counts)]
                    )
                ])
                fig.update_layout(
                    title="Prediction Distribution",
                    xaxis_title="Severity Level",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detailed results table
                st.subheader("📋 Detailed Results")
                st.dataframe(
                    results_df.style.format({
                        'confidence': '{:.2%}',
                        'no_dr': '{:.3f}',
                        'mild': '{:.3f}',
                        'moderate': '{:.3f}',
                        'severe': '{:.3f}',
                        'proliferative': '{:.3f}'
                    }),
                    use_container_width=True
                )

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="dr_predictions.csv",
                    mime="text/csv"
                )

    # ==================== TAB 3: ABOUT WITH XAI INFO ====================
    with tab3:
        st.header("About This System")

        st.markdown("""
        ### 🎯 Diabetic Retinopathy Classification System with XAI

        This system uses state-of-the-art deep learning (EfficientNet-B3) combined with 
        **Explainable AI (XAI)** techniques to provide transparent and interpretable predictions 
        for diabetic retinopathy detection.

        #### 🔬 XAI Methods Implemented:

        1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
           - Visualizes which regions of the retinal image influenced the prediction
           - Uses gradient information to weight feature maps
           - Fast and effective for clinical validation
           - Published in ICCV 2017

        2. **HiRes-CAM (High-Resolution CAM)**
           - Provides higher resolution attention maps
           - Better captures fine-grained lesions and microaneurysms
           - Improved localization accuracy over standard Grad-CAM
           - Especially useful for small DR features

        3. **Score-CAM**
           - Gradient-free visualization method
           - Uses forward pass scores instead of gradients
           - More stable in some cases
           - Useful for comparison with gradient methods

        4. **LIME (Local Interpretable Model-agnostic Explanations)**
           - Explains predictions by approximating the model locally
           - Shows which image regions contribute positively or negatively
           - Model-agnostic approach applicable to any classifier
           - Provides intuitive superpixel-based explanations

        #### 📊 Model Performance:
        - **Quadratic Weighted Kappa (QWK)**: 0.9488
        - **Architecture**: EfficientNet-B3 with custom classifier
        - **Input Size**: 224×224 pixels
        - **Classes**: 5 severity levels
          - Grade 0: No DR
          - Grade 1: Mild NPDR
          - Grade 2: Moderate NPDR
          - Grade 3: Severe NPDR
          - Grade 4: Proliferative DR

        #### 🏥 Clinical Benefits of XAI:
        - **Trust & Transparency**: Builds trust with healthcare professionals
        - **Validation**: Enables validation of AI predictions against clinical knowledge
        - **Bias Detection**: Helps identify potential model biases
        - **Decision Support**: Supports clinical decision-making without being a black box
        - **Regulatory Compliance**: Facilitates FDA/regulatory approval processes
        - **Education**: Helps train medical students and residents

        #### 🔍 Key Features:
        - Real-time prediction with confidence scores
        - Multiple XAI visualization methods
        - CLAHE preprocessing for fundus images
        - Batch processing capability
        - Detailed probability distributions
        - Clinical recommendations based on severity

        #### ⚙️ Technical Stack:
        - **Framework**: PyTorch 2.x
        - **Model**: EfficientNet-B3 (timm library)
        - **XAI**: pytorch-grad-cam, LIME
        - **Frontend**: Streamlit
        - **Visualization**: Plotly, Matplotlib

        #### 📚 References:
        - Grad-CAM: Selvaraju et al., ICCV 2017
        - HiRes-CAM: Draelos & Carin, 2020
        - LIME: Ribeiro et al., KDD 2016
        - EfficientNet: Tan & Le, ICML 2019

        #### ⚠️ Disclaimer:
        This tool is for research and educational purposes. It is designed to assist 
        healthcare professionals but should not replace clinical judgment. Always consult 
        with qualified ophthalmologists for clinical diagnosis and treatment decisions. 
        The XAI visualizations are intended to provide insight into model behavior and 
        should be interpreted by trained medical professionals.

        #### 👨‍⚕️ For Healthcare Providers:
        - Use XAI visualizations to verify model focus on clinically relevant features
        - Compare model attention with your clinical assessment
        - Report any discrepancies or unexpected model behavior
        - XAI can help explain predictions to patients

        #### 🔐 Privacy & Security:
        - All processing is done locally
        - No images are stored or transmitted
        - Model weights are loaded from local file system
        - Patient data remains on your device
        """)

        st.markdown("---")
        st.markdown("**Developed with ❤️ using PyTorch, Streamlit, and XAI libraries**")
        st.markdown("**Version**: 2.0 with Explainable AI Integration")

if __name__ == "__main__":
    main()
