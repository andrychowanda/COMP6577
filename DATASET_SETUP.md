# Dataset Setup Guide

## Quick Start

Create the following directory structure in the COMP6577 repository root:

```
COMP6577/
├── train/
│   ├── neutral/
│   ├── surprise/
│   ├── sad/
│   ├── happy/
│   ├── fear/
│   ├── disgust/
│   └── angry/
└── test/
    ├── neutral/
    ├── surprise/
    ├── sad/
    ├── happy/
    ├── fear/
    ├── disgust/
    └── angry/
```

## Dataset Requirements

### Video Format
- **File Extension**: `.mp4` (preferred) or other common video formats
- **Duration**: 3-16 seconds (variable length supported)
- **Resolution**: Any resolution (will be resized automatically)
- **Frame Rate**: Any frame rate (frames will be extracted)
- **Content**: Videos showing facial expressions of the labeled emotion

### Class Distribution

Ideally, have a balanced distribution across all 7 classes:
- neutral
- surprise
- sad
- happy
- fear
- disgust
- angry

### Recommended Split

- **Training**: 70-80% of total videos
- **Testing**: 20-30% of total videos

Example:
- If you have 700 videos total (100 per class)
- Place ~70 videos per class in `train/`
- Place ~30 videos per class in `test/`

## Creating the Directories

### Option 1: Manual Creation

```bash
# In the COMP6577 directory
mkdir -p train/{neutral,surprise,sad,happy,fear,disgust,angry}
mkdir -p test/{neutral,surprise,sad,happy,fear,disgust,angry}
```

### Option 2: Python Script

```python
import os

# Emotion classes
classes = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']

# Create train directories
for emotion_class in classes:
    os.makedirs(f'train/{emotion_class}', exist_ok=True)

# Create test directories
for emotion_class in classes:
    os.makedirs(f'test/{emotion_class}', exist_ok=True)

print("Dataset directories created successfully!")
```

## Populating the Dataset

### If you have existing videos

1. **Organize by emotion**: Group your videos by the emotion they express
2. **Split train/test**: Divide each emotion group into training and testing sets
3. **Copy to directories**: 
   ```bash
   # Example for neutral emotion
   cp my_neutral_videos/*.mp4 train/neutral/
   cp my_neutral_test_videos/*.mp4 test/neutral/
   ```

### If you need to collect videos

**Sources for emotion datasets:**

1. **FER2013** - Facial Expression Recognition dataset
2. **CK+** - Extended Cohn-Kanade dataset
3. **RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song
4. **AffectNet** - Large-scale facial expression dataset
5. **EmotioNet** - Emotion recognition dataset

**Note**: Ensure you have proper licenses/permissions to use these datasets.

### Naming Convention

While not strictly required, it's helpful to use descriptive filenames:

```
train/happy/happy_001.mp4
train/happy/happy_002.mp4
train/sad/sad_001.mp4
test/happy/happy_test_001.mp4
```

## Verifying Dataset Structure

Run this Python script to verify your dataset is set up correctly:

```python
import os
from pathlib import Path

classes = ['neutral', 'surprise', 'sad', 'happy', 'fear', 'disgust', 'angry']

print("Dataset Verification")
print("=" * 60)

for split in ['train', 'test']:
    print(f"\n{split.upper()} SET:")
    print("-" * 60)
    
    total_videos = 0
    for emotion_class in classes:
        class_dir = Path(split) / emotion_class
        
        if not class_dir.exists():
            print(f"  ❌ {emotion_class:12s}: Directory not found!")
            continue
        
        video_files = list(class_dir.glob('*.mp4'))
        num_videos = len(video_files)
        total_videos += num_videos
        
        status = "✓" if num_videos > 0 else "⚠"
        print(f"  {status} {emotion_class:12s}: {num_videos:4d} videos")
    
    print(f"\n  Total {split} videos: {total_videos}")

print("\n" + "=" * 60)
print("Verification complete!")
```

Expected output:
```
Dataset Verification
============================================================

TRAIN SET:
------------------------------------------------------------
  ✓ neutral     :   70 videos
  ✓ surprise    :   70 videos
  ✓ sad         :   70 videos
  ✓ happy       :   70 videos
  ✓ fear        :   70 videos
  ✓ disgust     :   70 videos
  ✓ angry       :   70 videos

  Total train videos: 490

TEST SET:
------------------------------------------------------------
  ✓ neutral     :   30 videos
  ✓ surprise    :   30 videos
  ✓ sad         :   30 videos
  ✓ happy       :   30 videos
  ✓ fear        :   30 videos
  ✓ disgust     :   30 videos
  ✓ angry       :   30 videos

  Total test videos: 210

============================================================
Verification complete!
```

## Minimum Requirements

For the notebook to run:
- At least 1 video in each class folder (both train and test)
- Recommended: At least 20-30 videos per class for meaningful results

## Face Detection Tips

The system will automatically detect and crop faces. For best results:

1. **Clear facial expressions**: Faces should be clearly visible
2. **Frontal view**: Best results with faces facing the camera
3. **Good lighting**: Avoid overly dark or bright videos
4. **Minimal occlusion**: Avoid glasses, masks, or other face obstructions
5. **One face per video**: System extracts the largest face if multiple faces present

## Troubleshooting

### "Dataset not found" error
- Verify `train/` and `test/` directories exist in the repository root
- Check that all 7 class subdirectories exist in both train and test
- Use the verification script above

### No faces detected
- Check video quality
- Ensure faces are visible and not too small
- Try videos with clearer, more frontal faces
- System will fall back to full frame if no face detected

### Class imbalance warnings
- Try to balance the number of videos across classes
- If imbalanced, pay attention to Macro-F1 score (handles imbalance better than accuracy)

## Data Augmentation

The notebook automatically applies data augmentation during training:
- Random horizontal flips
- Random rotations (±10°)
- Color jittering (brightness, contrast, saturation)

This helps improve model generalization even with limited data.

## Next Steps

Once your dataset is set up:
1. Verify structure using the verification script
2. Open `Emotion_Recognition_Model.ipynb`
3. Run all cells
4. Review the generated reports and visualizations

## Need Help?

If you encounter issues:
1. Check this guide thoroughly
2. Verify your directory structure matches exactly
3. Ensure video files are valid and readable
4. Check file permissions
5. Review error messages in the notebook
