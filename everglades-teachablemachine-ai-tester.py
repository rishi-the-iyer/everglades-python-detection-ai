import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

class_labels = ["python", "alligator", "misc"]  # Based on the order in Teachable Machine

# Load the Teachable Machine model
model = load_model('/Users/andy/code/python/keras_model.h5')

def classify_image(image_path):
    # Load image
    img = Image.open(image_path).resize((224, 224))
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img) / 255.0

    # Make prediction
    prediction = model.predict(img_array[np.newaxis, ...])
    return np.argmax(prediction, axis=1)[0]  # Assuming binary classification

# Example usage
# Directory containing test images
test_images_dir = '/Users/andy/Documents/ScienceProject/EvergladesPythons/feed/test'

# Iterate over each file in the directory
for filename in os.listdir(test_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image files
        file_path = os.path.join(test_images_dir, filename)
        result = classify_image(file_path)
        print(f'{filename} - Classified as: {class_labels[result]}')  # Add label mapping if needed
