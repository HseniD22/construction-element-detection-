# train_model.py
from utils.data_preprocessing import load_images_from_folder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import os
import joblib

# Load data
train_folder = 'data/train'
images, labels = load_images_from_folder(train_folder)

# Convert images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Flatten images for ML model
X = [img.flatten() for img in gray_images]
y = labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/construction_svm_model.pkl")
