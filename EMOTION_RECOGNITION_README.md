# Emotion Recognition Model - Complete Implementation

## Overview

This repository contains a comprehensive implementation of emotion recognition using three different approaches:

1. **Static Image** - Extracting single frames from videos
2. **Dynamic Image** - Temporal aggregation using rank pooling (based on [dynamic images for action recognition](https://github.com/tcvrick/dynamic-images-for-action-recognition))
3. **Video-based** - LSTM with CNN feature extraction

## Dataset Structure

### Required Directory Structure

```
COMP6577/
├── train/
│   ├── neutral/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── surprise/
│   │   └── *.mp4
│   ├── sad/
│   │   └── *.mp4
│   ├── happy/
│   │   └── *.mp4
│   ├── fear/
│   │   └── *.mp4
│   ├── disgust/
│   │   └── *.mp4
│   └── angry/
│       └── *.mp4
└── test/
    ├── neutral/
    ├── surprise/
    ├── sad/
    ├── happy/
    ├── fear/
    ├── disgust/
    └── angry/
```

### Dataset Requirements

- **Format**: MP4 video files
- **Classes**: 7 emotion classes
  - neutral
  - surprise
  - sad
  - happy
  - fear
  - disgust
  - angry
- **Video Duration**: 3-16 seconds (variable length supported)
- **Content**: Videos should contain faces expressing the labeled emotion

## Installation

### Required Packages

```bash
pip install torch torchvision opencv-python facenet-pytorch grad-cam lime scikit-learn tqdm matplotlib seaborn pandas pillow
```

### Optional (for better face detection)

```bash
pip install facenet-pytorch
```

If not installed, the system will fall back to OpenCV Haar Cascade face detection.

## Features

### 1. Face Detection and Cropping
- **MTCNN** (if facenet-pytorch is installed) for accurate face detection
- **Haar Cascade** fallback for environments without facenet-pytorch
- Automatic face cropping with configurable margins
- Focus on facial expressions for better emotion recognition

### 2. Three Model Approaches

#### Static Image Model
- Extracts a single frame (middle frame) from each video
- Uses standard CNN architectures (ResNet, VGG, MobileNet)
- Fast training and inference
- Baseline approach

#### Dynamic Image Model
- Generates a single "dynamic image" that encodes temporal information
- Uses rank pooling technique to aggregate temporal changes
- Maintains spatial resolution while capturing motion
- Same architecture as static models but with temporal information

#### Video Model
- Processes multiple frames (default: 16 frames)
- CNN feature extractor + LSTM for temporal modeling
- Captures full temporal dynamics
- More complex but potentially more accurate

### 3. Multiple Architectures
- **ResNet18** - Fast, good baseline
- **ResNet50** - More capacity, better accuracy
- **MobileNet V2** - Lightweight, efficient

### 4. Comprehensive Metrics

#### Top-1 Accuracy
- Overall classification accuracy on test set

#### Macro-F1 Score
- Important for handling class imbalance
- Average F1 score across all classes

#### Per-Class ROC-AUC
- ROC curves for each emotion class
- Area Under Curve (AUC) scores
- Helps identify which emotions are easiest/hardest to recognize

#### Confusion Matrix
- Visualizes which emotions are confused with each other
- Identifies systematic misclassification patterns

#### Classification Report
- Precision, recall, F1-score per class
- Support (number of samples) per class

### 5. Training Monitoring

#### Loss and Accuracy Curves
- Training and validation loss over epochs
- Training and validation accuracy over epochs
- Early stopping to prevent overfitting

#### Training Time Tracking
- Time per epoch
- Total training time
- Useful for comparing model efficiency

#### GPU Memory Monitoring
- Peak memory usage per epoch
- Helps optimize batch size and model selection

### 6. XAI (Explainable AI)

#### Grad-CAM Visualization
- Highlights which regions of the face the model focuses on
- Separate heatmap and overlay visualizations
- Applied to both static and dynamic image models
- Two overlays per sample as requested

#### Error Analysis
- Error buckets by true class
- Common misclassification pairs
- Error rate statistics

### 7. Dynamic Image Feature Visualization
- Shows how dynamic images are generated
- Displays: first frame, middle frame, last frame, and resulting dynamic image
- Helps understand temporal aggregation process

## Usage

### 1. Prepare Dataset

Place your video files in the required directory structure (see above).

### 2. Open the Notebook

```bash
jupyter notebook Emotion_Recognition_Model.ipynb
```

### 3. Run All Cells

The notebook is designed to run from top to bottom:

1. **Setup and Configuration** - Install packages, import libraries, configure parameters
2. **Face Detection** - Initialize face detection (MTCNN or Haar Cascade)
3. **Data Processing** - Define dataset classes and transformations
4. **Model Definitions** - Define all model architectures
5. **Training** - Train all models (static, dynamic, video) with all architectures
6. **Evaluation** - Comprehensive testing and metric calculation
7. **Visualization** - Generate all plots and visualizations
8. **Comparison** - Final comparison table and analysis
9. **XAI** - Grad-CAM and error analysis
10. **Save Results** - Export all results, models, and visualizations

### 4. Review Outputs

After running, check these directories:

- `outputs/` - JSON results and CSV comparison tables
- `models/` - Trained model weights (.pth files)
- `visualizations/` - All plots and figures

## Configuration

Edit the `Config` class in the notebook to customize:

```python
class Config:
    # Paths
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    
    # Training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 10
    
    # Video parameters
    MAX_FRAMES = 16
    FRAME_SAMPLING = 'uniform'  # or 'random'
    
    # Face detection
    FACE_MARGIN = 20
    MIN_FACE_SIZE = 20
    
    # Architectures to test
    ARCHITECTURES = ['resnet18', 'resnet50', 'mobilenet_v2']
```

## Outputs

### 1. Models
All trained models saved as `.pth` files in `models/` directory:
- `static_resnet18.pth`
- `dynamic_resnet50.pth`
- `video_mobilenet_v2.pth`
- etc.

### 2. Visualizations

#### Training Curves
- Loss (train and validation)
- Accuracy (train and validation)
- Training time per epoch
- GPU memory usage per epoch

#### Confusion Matrices
- Heatmap showing actual vs predicted classes
- Helps identify which emotions are confused

#### ROC Curves
- Per-class ROC curves
- All classes on same plot for comparison
- AUC scores in legend

#### Grad-CAM Heatmaps
- Original image
- Heatmap showing model attention
- Overlay of heatmap on image
- Two visualizations per sample

#### Dynamic Image Features
- Visualization of how dynamic images are generated
- Shows temporal aggregation process

### 3. Reports

#### Comparison Table (CSV)
- All models compared side-by-side
- Accuracy, Macro-F1, per-class AUC
- Exported to `outputs/model_comparison.csv`

#### Classification Reports
- Per-class precision, recall, F1-score
- Support (number of samples)
- Macro and weighted averages

#### Error Analysis
- Error rate by true class
- Most common misclassification pairs
- Error buckets

### 4. JSON Results
- `outputs/all_results.json` - All evaluation metrics
- `outputs/training_histories.json` - Training curves data

## Results Interpretation

### Choosing the Best Model

1. **Accuracy** - Overall correctness
2. **Macro-F1** - Performance across all classes (important for imbalanced data)
3. **Per-Class AUC** - Which model is best for each specific emotion
4. **Training Time** - Efficiency considerations
5. **GPU Memory** - Deployment constraints

### Understanding Confusion Matrices

- **Diagonal** - Correct predictions
- **Off-diagonal** - Confusions
- Common confusions:
  - Neutral ↔ Sad
  - Fear ↔ Surprise
  - Angry ↔ Disgust

### Interpreting Grad-CAM

- **Red/Yellow regions** - High importance for prediction
- **Blue regions** - Low importance
- Should focus on eyes, mouth, and eyebrows for emotions

## Troubleshooting

### Dataset Not Found
Ensure your directory structure matches the required format and that train/test directories exist.

### Out of Memory
- Reduce `BATCH_SIZE` in config
- Reduce `MAX_FRAMES` for video models
- Use smaller architecture (resnet18 instead of resnet50)

### Low Accuracy
- Check if faces are being detected properly
- Verify video quality
- Ensure correct labeling
- Try data augmentation
- Increase `NUM_EPOCHS`
- Try different architectures

### MTCNN Not Available
The notebook will automatically fall back to Haar Cascade. For better results, install:
```bash
pip install facenet-pytorch
```

## Citation

If you use this implementation, please cite:

```
Dynamic Images Implementation based on:
- Bilen, H., Fernando, B., Gavves, E., Vedaldi, A., & Gould, S. (2016).
  "Dynamic image networks for action recognition." CVPR 2016.
  https://github.com/tcvrick/dynamic-images-for-action-recognition
```

## License

This implementation is provided for educational purposes.

## Contact

For questions or issues, please refer to the repository's issue tracker.
