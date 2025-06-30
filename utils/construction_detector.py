# construction_detector.py
import cv2
import numpy as np
import joblib
from utils.visualization import draw_detections

def detect_from_image(image_path, model_path):
    # Load trained model
    model = joblib.load(model_path)

    # Load and preprocess image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(flat)[0]

    # For demo purposes, draw a fake box in the center
    h, w = image.shape[:2]
    x, y, box_w, box_h = w//4, h//4, w//2, h//2
    detections = [(x, y, box_w, box_h)]

    # Draw result
    result_img = draw_detections(image, detections, label=prediction)
    return result_img

if __name__ == "__main__":
    image_path = "data/test/sample1.jpg"
    model_path = "models/construction_svm_model.pkl"
    
    result = detect_from_image(image_path, model_path)

    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
