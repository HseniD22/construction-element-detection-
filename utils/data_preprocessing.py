# utils/data_preprocessing.py
import cv2
import os

def load_images_from_folder(folder_path, resize_dim=(224, 224)):
    images = []
    labels = []
    for label_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, resize_dim)
                images.append(img)
                labels.append(label_name)
    return images, labels

def convert_to_gray(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
