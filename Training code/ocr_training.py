"""
YOLOv8 Character Recognition Training Script
Indonesian License Plate Recognition Dataset
"""

import torch
from ultralytics import YOLO
import os

# Set random seeds for reproducibility
SEED = 42

def set_all_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# Configuration
DATA_YAML_PATH = 'data_char_recognition.yaml'
MODEL_NAME = 'yolov8n.pt'  # Using nano model for faster training
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT_NAME = 'char_recognition'

if __name__ == '__main__':
    # Check CUDA availability
    print("=" * 60)
    print("YOLO Character Recognition Training Setup")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device being used: {torch.cuda.get_device_name(0)}")
    else:
        print("Device being used: CPU")
    print("=" * 60)

    # Load pre-trained YOLOv8 model
    print(f"\nLoading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Move model to appropriate device
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(target_device)

    # Verify device
    first_param_device = next(model.model.parameters()).device
    print(f"✅ Model device: {first_param_device}\n")

    # Start training
    print("=" * 60)
    print("Starting YOLOv8 Character Recognition Training...")
    print("=" * 60)
    print(f"Dataset: {DATA_YAML_PATH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Number of classes: 36 (0-9, A-Z)")
    print("=" * 60)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        device=0 if torch.cuda.is_available() else 'cpu',
        amp=True,  # Automatic Mixed Precision
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True,
        seed=SEED,
        deterministic=True,
        # Optimization parameters
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation parameters (moderate for character recognition)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,  # Rotation
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,  # No vertical flip for characters
        fliplr=0.0,  # No horizontal flip for characters
        mosaic=1.0,
        mixup=0.0,
    )

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"Results saved to: runs/detect/{PROJECT_NAME}")
    print("=" * 60)

    # Validate the best model
    print("\n" + "=" * 60)
    print("Validating best model...")
    print("=" * 60)
    metrics = model.val()

    print("\n" + "=" * 60)
    print("Validation Metrics:")
    print("=" * 60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 60)

    print("\n✅ Character recognition model training completed successfully!")
    print(f"Best model saved at: runs/detect/{PROJECT_NAME}/weights/best.pt")
