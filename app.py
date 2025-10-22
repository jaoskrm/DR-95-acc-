# ================================================================
# DIABETIC RETINOPATHY WEB APP - STREAMLIT
# Single + Batch Image Testing with State-of-the-Art Model
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

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="DR Classifier",
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
        # Add weights_only=False to fix PyTorch 2.6+ loading
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
        'confidence': probabilities[predicted_class].item()
    }

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

def process_batch(model, images, device='cuda'):
    """Process multiple images"""
    results = []
    progress_bar = st.progress(0)
    
    for idx, (name, img) in enumerate(images):
        try:
            result = predict_image(model, img, device, apply_preprocessing=True)
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
                'no_dr': 0, 'mild': 0, 'moderate': 0, 'severe': 0, 'proliferative': 0
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
    st.title("👁️ Diabetic Retinopathy Classifier")
    st.markdown("### State-of-the-Art DR Detection (QWK: 0.9488)")
    
    # Mode selection
    tab1, tab2, tab3 = st.tabs(["🖼️ Single Image", "📁 Batch Processing", "ℹ️ About"])
    
    # ==================== TAB 1: SINGLE IMAGE ====================
    with tab1:
        st.header("Single Image Prediction")
        
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
                confidence_col1, confidence_col2, confidence_col3 = st.columns([1, 2, 1])
                with confidence_col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.2f}%",
                        delta=None
                    )
                
                # Recommendation
                st.info(get_recommendation(result['label']))
                
                # Probability chart
                st.plotly_chart(
                    create_probability_chart(result['probabilities']),
                    use_container_width=True
                )
                
                # Detailed probabilities
                st.subheader("Detailed Probabilities")
                prob_df = pd.DataFrame({
                    'Class': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
                    'Probability (%)': [f"{p*100:.2f}" for p in result['probabilities']]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 2: BATCH PROCESSING ====================
    with tab2:
        st.header("Batch Image Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple fundus images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple retinal fundus photographs"
        )
        
        if uploaded_files and model is not None:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                # Load images
                images = [(f.name, Image.open(f).convert('RGB')) for f in uploaded_files]
                
                # Process
                with st.spinner(f"Processing {len(images)} images..."):
                    results_df = process_batch(model, images, device)
                
                st.success(f"✅ Processed {len(results_df)} images!")
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    no_dr_count = (results_df['prediction'] == 'No DR').sum()
                    st.metric("No DR", no_dr_count)
                
                with col2:
                    mild_count = (results_df['prediction'] == 'Mild').sum()
                    st.metric("Mild", mild_count)
                
                with col3:
                    moderate_count = (results_df['prediction'] == 'Moderate').sum()
                    st.metric("Moderate", moderate_count)
                
                with col4:
                    severe_count = (results_df['prediction'] == 'Severe').sum()
                    st.metric("Severe", severe_count)
                
                with col5:
                    prolif_count = (results_df['prediction'] == 'Proliferative DR').sum()
                    st.metric("Proliferative", prolif_count)
                
                # Distribution chart
                pred_counts = results_df['prediction'].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Distribution of Predictions",
                    color_discrete_sequence=['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("📋 Detailed Results")
                
                # Format confidence as percentage
                display_df = results_df.copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(
                    display_df[['filename', 'prediction', 'confidence']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="dr_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ==================== TAB 3: ABOUT ====================
    with tab3:
        st.header("About This Model")
        
        st.markdown("""
        ### 🎯 Model Performance
        
        **Architecture:** EfficientNet-B3 with custom classification head
        
        **Training Dataset:** 7,324 fundus images (APTOS 2019 enhanced)
        
        **Performance Metrics:**
        - **QWK (Quadratic Weighted Kappa):** 0.9488 ⭐
        - **Accuracy:** 92.36%
        - **Macro F1-Score:** 0.8502
        
        ### 📊 Per-Class Performance
        
        | Class | Precision | Recall | F1-Score |
        |-------|-----------|--------|----------|
        | No DR | 99.1% | 99.1% | 0.991 |
        | Mild | 86.1% | 83.8% | 0.849 |
        | Moderate | 87.0% | 94.0% | 0.904 |
        | Severe | 80.9% | 65.5% | 0.724 |
        | Proliferative | 83.3% | 73.9% | 0.783 |
        
        ### 🔬 Preprocessing
        
        - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** in LAB color space
        - Enhances contrast and reveals fine details in fundus images
        - Improves lesion detection accuracy
        
        ### ⚠️ Important Notes
        
        - This is a screening tool, not a diagnostic system
        - Results should be verified by qualified ophthalmologists
        - Severe/Proliferative DR detection has 65-74% recall
        - Best used for population screening and triage
        
        ### 📚 References
        
        - APTOS 2019 Blindness Detection Dataset
        - EfficientNet-B3 architecture (Tan & Le, 2019)
        - QWK loss for ordinal classification
        
        ### 👨‍💻 Model Details
        
        - **Parameters:** 11.5M trainable parameters
        - **Training Time:** ~4 hours on RTX 4080
        - **Image Size:** 224×224 pixels
        - **Device:** GPU-accelerated (CUDA) inference
        """)

if __name__ == "__main__":
    main()
