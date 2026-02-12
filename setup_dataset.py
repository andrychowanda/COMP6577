#!/usr/bin/env python3
"""
Setup Dataset Directories for Emotion Recognition

This script creates the required directory structure for the emotion recognition dataset.
Run this before adding your video files.
"""

import os
from pathlib import Path

# Emotion classes
CLASSES = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']

def setup_directories():
    """Create train and test directories with class subdirectories"""
    print("\n" + "=" * 70)
    print("EMOTION RECOGNITION - DATASET SETUP")
    print("=" * 70)
    print("\nCreating directory structure...\n")
    
    created_dirs = []
    existing_dirs = []
    
    for split in ['train', 'test']:
        split_path = Path(split)
        
        # Create split directory
        if not split_path.exists():
            split_path.mkdir()
            created_dirs.append(str(split_path))
        else:
            existing_dirs.append(str(split_path))
        
        # Create class subdirectories
        for emotion_class in CLASSES:
            class_dir = split_path / emotion_class
            if not class_dir.exists():
                class_dir.mkdir()
                created_dirs.append(str(class_dir))
                print(f"  ✓ Created: {class_dir}/")
            else:
                existing_dirs.append(str(class_dir))
                print(f"  ⚠ Exists:  {class_dir}/")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if created_dirs:
        print(f"\n✓ Created {len(created_dirs)} new directories")
    
    if existing_dirs:
        print(f"⚠ {len(existing_dirs)} directories already existed")
    
    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE")
    print("=" * 70)
    print("""
COMP6577/
├── train/
│   ├── neutral/      ← Add neutral emotion videos here
│   ├── surprise/     ← Add surprise emotion videos here
│   ├── sad/          ← Add sad emotion videos here
│   ├── happy/        ← Add happy emotion videos here
│   ├── fear/         ← Add fear emotion videos here
│   ├── disgust/      ← Add disgust emotion videos here
│   └── angry/        ← Add angry emotion videos here
└── test/
    ├── neutral/      ← Add neutral test videos here
    ├── surprise/     ← Add surprise test videos here
    ├── sad/          ← Add sad test videos here
    ├── happy/        ← Add happy test videos here
    ├── fear/         ← Add fear test videos here
    ├── disgust/      ← Add disgust test videos here
    └── angry/        ← Add angry test videos here
    """)
    
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Add your MP4 video files to the appropriate folders
   - Training videos go in train/<emotion>/
   - Testing videos go in test/<emotion>/

2. Verify your dataset:
   python3 verify_dataset.py

3. Run the emotion recognition notebook:
   jupyter notebook Emotion_Recognition_Model.ipynb

For more information, see DATASET_SETUP.md
    """)
    print("=" * 70 + "\n")

if __name__ == '__main__':
    setup_directories()
