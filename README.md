# COMP6577

This repository contains various machine learning and deep learning projects.

## Recent Addition: Emotion Recognition Model 🎭

A comprehensive emotion recognition system comparing three approaches:
- **Static Images** - Single frame extraction
- **Dynamic Images** - Temporal aggregation using rank pooling
- **Video-based** - LSTM with CNN features

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your dataset** (see [DATASET_SETUP.md](DATASET_SETUP.md)):
   ```bash
   mkdir -p train/{neutral,surprise,sad,happy,fear,disgust,angry}
   mkdir -p test/{neutral,surprise,sad,happy,fear,disgust,angry}
   # Add your MP4 videos to each folder
   ```

3. **Verify dataset**:
   ```bash
   python3 verify_dataset.py
   ```

4. **Run the notebook**:
   ```bash
   jupyter notebook Emotion_Recognition_Model.ipynb
   ```

### Features

✅ Face detection and cropping (MTCNN/Haar Cascade)  
✅ Three model approaches with multiple architectures  
✅ Comprehensive metrics (Accuracy, Macro-F1, ROC-AUC)  
✅ XAI visualizations (Grad-CAM)  
✅ Training/validation tracking  
✅ GPU memory monitoring  
✅ Error analysis and comparison tables  

### Documentation

- 📘 [Complete Guide](EMOTION_RECOGNITION_README.md) - Detailed usage and features
- 📁 [Dataset Setup](DATASET_SETUP.md) - How to organize your data
- 📦 [Requirements](requirements.txt) - All dependencies
- 📓 [Notebook](Emotion_Recognition_Model.ipynb) - Main implementation

---

## Previous Projects

### Datasets
- Iris Dataset: https://archive.ics.uci.edu/ml/datasets/Iris
- Singapore AirBnB: https://www.kaggle.com/jojoker/singapore-airbnb
- Titanic Dataset: https://www.kaggle.com/c/titanic-dataset/data
- Insurance Dataset: https://www.kaggle.com/gloriousc/insurance-forecast-by-using-linear-regression/data