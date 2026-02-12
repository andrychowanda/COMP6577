#!/usr/bin/env python3
"""
Dataset Verification Script for Emotion Recognition
Run this script to verify your dataset is properly set up before running the notebook.
"""

import os
from pathlib import Path
from collections import defaultdict

# Emotion classes
CLASSES = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']
SPLITS = ['train', 'test']

def check_dataset():
    """Verify dataset structure and contents"""
    print("\n" + "=" * 80)
    print("EMOTION RECOGNITION DATASET VERIFICATION")
    print("=" * 80)
    
    issues = []
    stats = defaultdict(dict)
    
    for split in SPLITS:
        print(f"\n{split.upper()} SET:")
        print("-" * 80)
        
        split_path = Path(split)
        
        if not split_path.exists():
            print(f"  ❌ ERROR: '{split}/' directory not found!")
            issues.append(f"Missing '{split}/' directory")
            continue
        
        total_videos = 0
        
        for emotion_class in CLASSES:
            class_dir = split_path / emotion_class
            
            if not class_dir.exists():
                print(f"  ❌ {emotion_class:12s}: Directory not found!")
                issues.append(f"Missing '{split}/{emotion_class}/' directory")
                stats[split][emotion_class] = 0
                continue
            
            # Count video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
                video_files.extend(list(class_dir.glob(ext)))
            
            num_videos = len(video_files)
            total_videos += num_videos
            stats[split][emotion_class] = num_videos
            
            # Status indicators
            if num_videos == 0:
                status = "⚠️ "
                message = "No videos found!"
                issues.append(f"No videos in '{split}/{emotion_class}/'")
            elif num_videos < 10:
                status = "⚠️ "
                message = f"{num_videos:4d} videos (recommended: 20+)"
            else:
                status = "✓ "
                message = f"{num_videos:4d} videos"
            
            print(f"  {status} {emotion_class:12s}: {message}")
        
        print(f"\n  {'TOTAL':14s}: {total_videos} videos in {split} set")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not issues:
        print("\n✅ Dataset structure looks good!")
        
        # Check balance
        print("\nClass Distribution:")
        print("-" * 80)
        
        for split in SPLITS:
            if split in stats and stats[split]:
                counts = list(stats[split].values())
                min_count = min(counts)
                max_count = max(counts)
                avg_count = sum(counts) / len(counts)
                
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"\n{split.upper()}:")
                print(f"  Min:     {min_count}")
                print(f"  Max:     {max_count}")
                print(f"  Average: {avg_count:.1f}")
                print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
                
                if imbalance_ratio > 3:
                    print(f"  ⚠️  Warning: Significant class imbalance detected!")
                    print(f"     Consider balancing the dataset or using weighted loss.")
        
        print("\n" + "=" * 80)
        print("✅ READY TO RUN: Open Emotion_Recognition_Model.ipynb")
        print("=" * 80)
        
    else:
        print("\n❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n" + "=" * 80)
        print("REQUIRED DIRECTORY STRUCTURE:")
        print("=" * 80)
        print("""
COMP6577/
├── train/
│   ├── neutral/
│   │   └── *.mp4
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
        """)
        
        print("\nSee DATASET_SETUP.md for detailed instructions.")
        print("=" * 80)
    
    print()
    return len(issues) == 0

if __name__ == '__main__':
    success = check_dataset()
    exit(0 if success else 1)
