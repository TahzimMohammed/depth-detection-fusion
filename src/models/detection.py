"""
Object Detection using YOLOv8
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Union
from PIL import Image

class ObjectDetector:
    """
    YOLOv8-based object detector for depth fusion.
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        Initialize the object detector.
        """
        print(f"Initializing YOLOv8 detector: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        print(f"✓ Detector loaded (confidence threshold: {conf_threshold})")
        
    def detect(self, image: Union[np.ndarray, Image.Image, str]) -> List[Dict]:
        # Run object detection on an image.

        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # Parse detections into our format
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            # Extract information for each detected object
            for i in range(len(boxes)):
                # Bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Class ID and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get human-readable class name
                class_name = self.model.names[class_id]
                
                # Store detection
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        return detections
    
    def filter_by_classes(self, detections: List[Dict], 
                         class_names: List[str]) -> List[Dict]:
  
        # Filter detections to only include specific object classes.

        return [det for det in detections if det['class_name'] in class_names]
    
    def filter_by_size(self, detections: List[Dict], 
                      min_size: int = 20, 
                      max_size: int = None) -> List[Dict]:
        
        # Filter detections by bounding box size.
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Check minimum size
            if width < min_size or height < min_size:
                continue
            
            # Check maximum size if specified
            if max_size is not None:
                if width > max_size or height > max_size:
                    continue
            
            filtered.append(det)
        
        return filtered


def test_detector():

    # Testing function to verify the detector works.
    print("\n" + "="*60)
    print("Testing YOLOv8 Object Detector")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = ObjectDetector(model_name='yolov8n.pt', conf_threshold=0.5)
    
    print("\nDetector initialized successfully!")
    print(f"Available classes: {len(detector.model.names)} categories")
    print(f"Example classes: {list(detector.model.names.values())[:10]}")
    
    print("\n✓ Detection module is ready!")
    print("\nTo use in your code:")
    print("  from detection import ObjectDetector")
    print("  detector = ObjectDetector()")
    print("  detections = detector.detect('image.jpg')")
    print("\n" + "="*60)


if __name__ == '__main__':
    test_detector()