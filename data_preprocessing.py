# src/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.classes = []  # Will be populated from folder names
        
    def load_data(self, data_path):
        images = []
        labels = []
        class_names = []
        
        # Get all class folders
        classes = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
        self.classes = classes
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(data_path, class_name)
            
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    images.append(img)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels), classes
    
    def preprocess_images(self, images):
        # Normalize pixel values
        images = images.astype('float32') / 255.0
        
        # Data augmentation (optional)
        return images
    
    def prepare_data(self, data_path, test_size=0.2, val_size=0.2):
        X, y, classes = self.load_data(data_path)
        X = self.preprocess_images(X)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=len(classes))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=np.argmax(y_train, axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, classes