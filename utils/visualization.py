# utils/visualization.py
import cv2

def draw_detections(image, detections, label="Object", color=(0, 255, 0)):
    """
    Draw bounding boxes on the image.
    
    Parameters:
    - image: The image on which to draw
    - detections: List of (x, y, w, h) tuples
    - label: Label to display
    - color: Box color
    """
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image
