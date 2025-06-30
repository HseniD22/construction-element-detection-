# utils/model_utils.py
import cv2

def load_haar_model(model_path):
    """Load Haar Cascade Classifier model from .xml file"""
    model = cv2.CascadeClassifier(model_path)
    if model.empty():
        raise ValueError("Failed to load Haar model. Check the path.")
    return model

def save_model(model, save_path):
    """Placeholder for saving a model (not needed for Haar, but useful for ML models)"""
    import joblib
    joblib.dump(model, save_path)

def load_ml_model(model_path):
    """Load saved machine learning model"""
    import joblib
    return joblib.load(model_path)
