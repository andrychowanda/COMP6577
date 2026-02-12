#!/usr/bin/env python3
"""
Example: Using Trained Emotion Recognition Models for Inference

This script demonstrates how to load a trained model and make predictions
on new videos.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration (should match training config)
CLASSES = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']
IMG_SIZE = 224
MAX_FRAMES = 16

# Face detection (simplified for example)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Transform (same as test transform)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_face(frame):
    """Simple face detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

def extract_frames(video_path, max_frames=None):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    if max_frames and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return frames

def generate_dynamic_image(frames):
    """Generate dynamic image from frames"""
    if len(frames) == 0:
        return None
    
    frames = np.array(frames).astype(np.float32)
    num_frames = len(frames)
    
    # Weighted sum with increasing weights
    weights = np.arange(1, num_frames + 1) / num_frames
    weights = weights / weights.sum()
    
    dynamic_img = np.zeros_like(frames[0])
    for i, frame in enumerate(frames):
        dynamic_img += weights[i] * frame
    
    return dynamic_img.astype(np.uint8)

def load_static_model(model_path, arch='resnet18'):
    """Load a trained static/dynamic image model"""
    if arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    elif arch == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_static(model, video_path):
    """Predict emotion using static image model"""
    # Extract middle frame
    frames = extract_frames(video_path, max_frames=1)
    if not frames:
        return None, None
    
    # Detect face
    frame = detect_face(frames[0])
    
    # Transform and predict
    img = Image.fromarray(frame)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    emotion = CLASSES[pred_idx.item()]
    conf = confidence.item()
    
    return emotion, conf

def predict_dynamic(model, video_path):
    """Predict emotion using dynamic image model"""
    # Extract frames
    frames = extract_frames(video_path, max_frames=MAX_FRAMES)
    if not frames:
        return None, None
    
    # Detect faces in all frames
    face_frames = [detect_face(f) for f in frames]
    
    # Generate dynamic image
    dynamic_img = generate_dynamic_image(face_frames)
    
    # Transform and predict
    img = Image.fromarray(dynamic_img)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    emotion = CLASSES[pred_idx.item()]
    conf = confidence.item()
    
    return emotion, conf

# Example usage
if __name__ == '__main__':
    import sys
    
    print("Emotion Recognition - Inference Example")
    print("=" * 60)
    
    # Check if model and video are provided
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python3 inference_example.py <model_path> <video_path>")
        print("\nExample:")
        print("  python3 inference_example.py models/static_resnet18.pth test/happy/video1.mp4")
        print("\nModel types:")
        print("  - static_*.pth    : Use predict_static()")
        print("  - dynamic_*.pth   : Use predict_dynamic()")
        print("  - video_*.pth     : Requires Video model class")
        sys.exit(1)
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    
    # Check files exist
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    # Determine architecture from filename
    arch = 'resnet18'
    if 'resnet50' in model_path:
        arch = 'resnet50'
    elif 'mobilenet' in model_path:
        arch = 'mobilenet_v2'
    
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Architecture: {arch}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_static_model(model_path, arch=arch)
    
    # Predict
    print("Processing video...")
    
    if 'dynamic' in model_path:
        emotion, confidence = predict_dynamic(model, video_path)
    else:
        emotion, confidence = predict_static(model, video_path)
    
    # Results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    if emotion:
        print(f"Emotion:    {emotion.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print()
        
        # Show all probabilities
        print("All class probabilities:")
        frames = extract_frames(video_path, max_frames=1)
        if 'dynamic' in model_path:
            frames = extract_frames(video_path, max_frames=MAX_FRAMES)
            face_frames = [detect_face(f) for f in frames]
            img_array = generate_dynamic_image(face_frames)
        else:
            img_array = detect_face(frames[0])
        
        img = Image.fromarray(img_array)
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        for i, (class_name, prob) in enumerate(zip(CLASSES, probs)):
            bar = '█' * int(prob * 50)
            print(f"  {class_name:10s}: {prob:6.2%} {bar}")
    else:
        print("Error: Could not process video")
    
    print("=" * 60)
