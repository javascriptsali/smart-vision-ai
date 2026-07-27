"""
YOLO Object Detection Module
Handles model loading, inference, and result processing.
"""

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict
import os


class ObjectDetector:
    """YOLOv8-based object detector."""
    
    def __init__(self, model_name: str = 'yolov8n.pt'):
        """
        Initialize the detector with a YOLOv8 model.
        
        Args:
            model_name: Model variant (n=nano, s=small, m=medium, l=large, x=xlarge)
        """
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.model_name = model_name
        print("Model loaded successfully!")
    
    def detect_image(self, image: Image.Image, confidence: float = 0.25, 
                    iou_threshold: float = 0.45) -> Dict:
        """
        Detect objects in a PIL Image.
        
        Args:
            image: PIL Image object
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary with detections, annotated image, and analytics
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = self.model(img_array, conf=confidence, iou=iou_threshold, verbose=False)
        
        # Process results
        detections = self._process_results(results[0])
        
        # Get annotated image
        annotated = results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        
        # Generate analytics
        analytics = self._generate_analytics(detections)
        
        return {
            'detections': detections,
            'annotated_image': annotated_pil,
            'analytics': analytics,
            'total_objects': len(detections)
        }

    def detect_video(self, video_path: str, confidence: float = 0.25,
                     frame_skip: int = 5) -> Dict:
        """
        Detect objects in a video file.
        Processes every frame_skip frames for performance.
        
        Args:
            video_path: Path to the video file
            confidence: Minimum confidence threshold
            frame_skip: Process every Nth frame (higher = faster but less accurate)
            
        Returns:
            Dictionary with video analytics and detections
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        all_detections = []
        frame_count = 0
        processed_frames = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame_skip frames
            if frame_count % frame_skip == 0:
                results = self.model(frame, conf=confidence, verbose=False)
                detections = self._process_results(results[0])
                all_detections.extend(detections)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Generate analytics from all detections
        analytics = self._generate_analytics(all_detections)
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'detections': all_detections,
            'analytics': analytics,
            'total_objects': len(all_detections)
        }
    
    def _process_results(self, result) -> List[Dict]:
        """
        Process YOLO results into structured format.
        
        Args:
            result: YOLO result object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return detections
    
    def _generate_analytics(self, detections: List[Dict]) -> Dict:
        """
        Generate analytics from detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Analytics dictionary
        """
        if not detections:
            return {
                'class_counts': {},
                'avg_confidence': 0,
                'unique_classes': 0,
                'most_common': None
            }
        
        # Count by class
        class_counts = {}
        confidences = []
        
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            confidences.append(det['confidence'])
        
        # Find most common
        most_common = max(class_counts.items(), key=lambda x: x[1]) if class_counts else None
        
        return {
            'class_counts': class_counts,
            'avg_confidence': sum(confidences) / len(confidences),
            'unique_classes': len(class_counts),
            'most_common': most_common
        }


# Singleton instance for efficiency
_detector_instance = None


def get_detector(model_name: str = 'yolov8n.pt') -> ObjectDetector:
    """
    Get or create a singleton detector instance.
    
    Args:
        model_name: Model variant to use
        
    Returns:
        ObjectDetector instance
    """
    global _detector_instance
    if _detector_instance is None or _detector_instance.model_name != model_name:
        _detector_instance = ObjectDetector(model_name)
    return _detector_instance


# Quick test function
def test_detector():
    """Test the detector with a sample image."""
    print("\n=== Testing Object Detector ===\n")
    
    # Initialize detector
    detector = get_detector('yolov8n.pt')
    
    # Create a simple test image (or load from file)
    # For testing, we'll create a blank image
    test_image = Image.new('RGB', (640, 480), color='blue')
    
    # Run detection
    results = detector.detect_image(test_image, confidence=0.25)
    
    print(f"\nTotal objects detected: {results['total_objects']}")
    print(f"Analytics: {results['analytics']}")
    print("\n✅ Detector test completed successfully!")


if __name__ == "__main__":
    test_detector()