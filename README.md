# Diabetic Retinopathy Detection with Explainable AI

A diabetic retinopathy classification project built on EfficientNet-B3 with progressively advanced training pipelines: standard training, training with explainable AI (XAI) integration, and an optimized pipeline combining SAM, SWA, CORAL, and mixup/cutmix augmentation. Streamlit applications provide single-image and batch (ZIP) testing, with XAI visualizations powered by GradCAM, HiResCAM, ScoreCAM, LIME, and Captum Occlusion.

## Tech Stack

- Python
- PyTorch
- timm (EfficientNet-B3)
- Albumentations
- Streamlit
- pytorch-grad-cam (GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM)
- LIME
- Captum
- Scikit-image / OpenCV

## Features

- 5-class diabetic retinopathy severity classification with EfficientNet-B3
- Three training pipelines:
  - Standard training (`model.py`): per-epoch confusion matrix logging, 50 epochs, LR 3e-4
  - XAI-integrated training (`model_with_xai.py`): GradCAM, HiResCAM, ScoreCAM, and LIME explanations
  - Optimized training (`model_optimized.py`): SAM optimizer, SWA (starting at epoch 120), CORAL, mixup/cutmix, 150 epochs
- Streamlit applications:
  - Single-image and batch testing with ZIP upload (`app.py`)
  - XAI-enabled app with GradCAM, HiResCAM, ScoreCAM, and LIME views (`app_with_xai.py`)
  - Advanced app with GradCAMPlusPlus, GradCAM, Captum Occlusion, and LIME (`dr_detection_app.py`)
- GPU memory management via expandable segments
- Per-class metrics and confusion matrices

## Project Structure

```
DR-Project/
├── model.py               # Standard EfficientNet-B3 training
├── model_with_xai.py      # Training with XAI explanations
├── model_optimized.py     # Optimized training (SAM/SWA/CORAL/mixup/cutmix)
├── app.py                 # Streamlit: single + batch (ZIP) testing
├── app_with_xai.py        # Streamlit with XAI visualizations
└── dr_detection_app.py    # Advanced Streamlit app (CAM/Occlusion/LIME)
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Retinal fundus image dataset organized by severity classes

### Installation

There is no requirements.txt; install the core dependencies directly:

```bash
pip install torch torchvision timm albumentations streamlit pytorch-grad-cam lime captum scikit-image opencv-python
```

### Running

Train the optimized model:

```bash
python model_optimized.py
```

Launch the base app:

```bash
streamlit run app.py
```

Launch the XAI app:

```bash
streamlit run app_with_xai.py
```

## Models Used

- EfficientNet-B3 (timm) with custom classification heads (`EfficientNetDR`, `EfficientNetDR_V2`)

## Notes

- Training data is expected at `D:\studies\clg\cao\data\colored_images`.
- The repository does not include a requirements.txt or pretrained weights; dependencies must be installed from the imports listed above.
- This tool is for educational and research purposes only, not a substitute for professional medical diagnosis.
