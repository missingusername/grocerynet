import os

# disabling warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set the path to the input and output directories
in_path = 'in'
data_path = os.path.join(in_path, 'combined_dataset')
output_path = 'out'

# Define the filename for the saved data and model
saved_data_file = os.path.join(output_path, 'saved_data.pkl')
model_file = os.path.join(output_path, 'grocery_resnet50.h5')
label_binarizer_file = os.path.join(output_path, 'label_binarizer.pkl')

# Function to load the saved data file
def load_data(saved_data_file):
    with open(saved_data_file, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform prediction
def predict(image_path):
    # Load the model
    model = load_model(model_file)

    # Load the LabelBinarizer
    with open(label_binarizer_file, 'rb') as f:
        lb = pickle.load(f)

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Perform prediction
    predictions = model.predict(image)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_confidences = predictions[0][top_5_indices]

    # Get the labels for the top 5 predictions
    top_5_labels = [lb.classes_[i] for i in top_5_indices]

    print('Top 5 Predictions:')
    for label, confidence in zip(top_5_labels, top_5_confidences):
        print(f'{label}: {confidence:.2f}')

if __name__ == '__main__':
    image_path = os.path.join(output_path, 'juice2.jpg')

    predict(image_path)
