"""
YOLOv8 Character Recognition Inference Script
Test the trained character recognition model on license plate images
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

# Configuration
MODEL_PATH = "runs/detect/char_recognition3/weights/best.pt"  # Path to trained model
TEST_IMAGE_DIR = "Indonesian License Plate Recognition Dataset/images/test"  # Test images directory
OUTPUT_DIR = "char_recognition_results"  # Output directory for results
CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold for detections

# Class names (0-9, A-Z)
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def sort_boxes_left_to_right(boxes):
    """Sort bounding boxes from left to right based on x-coordinate"""
    if len(boxes) == 0:
        return []
    
    # Get x-coordinates (center x)
    x_coords = [(box.xyxy[0][0].item(), box) for box in boxes]
    # Sort by x-coordinate
    sorted_boxes = [box for _, box in sorted(x_coords, key=lambda x: x[0])]
    return sorted_boxes

def recognize_characters(image_path, model, conf_threshold=0.25):
    """
    Recognize characters in a license plate image
    
    Args:
        image_path: Path to the image
        model: YOLO model
        conf_threshold: Confidence threshold
        
    Returns:
        recognized_text: String of recognized characters
        annotated_image: Image with bounding boxes and labels
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return "", None
    
    # Run inference
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # Get detections
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print(f"No characters detected in {image_path}")
        return "", img
    
    # Sort boxes from left to right
    sorted_boxes = sort_boxes_left_to_right(boxes)
    
    # Extract characters
    recognized_chars = []
    annotated_img = img.copy()
    
    for box in sorted_boxes:
        # Get class ID and confidence
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        # Get character
        char = CLASS_NAMES[cls_id]
        recognized_chars.append(char)
        
        # Draw bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        label = f"{char} {conf:.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    recognized_text = ''.join(recognized_chars)
    return recognized_text, annotated_img

def main():
    """Main inference function"""
    print("=" * 60)
    print("YOLOv8 Character Recognition Inference")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_char_recognition.py")
        return
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get test images
    test_images = []
    if os.path.exists(TEST_IMAGE_DIR):
        test_images = list(Path(TEST_IMAGE_DIR).glob("*.jpg")) + \
                     list(Path(TEST_IMAGE_DIR).glob("*.png"))
    
    if len(test_images) == 0:
        print(f"⚠️  No test images found in {TEST_IMAGE_DIR}")
        print("Please provide test images or update TEST_IMAGE_DIR path")
        return
    
    print(f"Found {len(test_images)} test images")
    print("=" * 60)
    
    # Process each image
    results_summary = []
    
    for idx, img_path in enumerate(test_images[:20], 1):  # Process first 20 images
        print(f"\n[{idx}/{min(20, len(test_images))}] Processing: {img_path.name}")
        
        # Recognize characters
        recognized_text, annotated_img = recognize_characters(
            img_path, model, CONFIDENCE_THRESHOLD
        )
        
        if recognized_text:
            print(f"✅ Recognized: {recognized_text}")
            
            # Save annotated image
            output_path = os.path.join(OUTPUT_DIR, f"result_{img_path.name}")
            cv2.imwrite(output_path, annotated_img)
            
            results_summary.append({
                'image': img_path.name,
                'text': recognized_text
            })
        else:
            print(f"⚠️  No characters detected")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Recognition Summary")
    print("=" * 60)
    for result in results_summary:
        print(f"{result['image']}: {result['text']}")
    
    print("\n" + "=" * 60)
    print(f"✅ Inference complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
