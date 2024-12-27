import os

# disabling warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from codecarbon import EmissionsTracker

# Set the path to the input and output directories
in_path = 'in'
output_path = 'out'

# Define the filename for the saved data and model
model_file = os.path.join(output_path, '5', 'grocerynet50_5epochs.keras')
label_binarizer_file = os.path.join(output_path, 'label_binarizer.pkl')

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform prediction
def predict(image_path):
    tracker = EmissionsTracker(
        project_name="ResNet50 Prediction",
        experiment_id="Prediction",
        output_dir=output_path,
        output_file="prediction_emissions.csv"
    )

    tracker.start()

    # Load the model
    tracker.start_task('load model')
    model = load_model(model_file)
    tracker.stop_task()

    # Load the LabelBinarizer
    tracker.start_task('load label binarizer')
    with open(label_binarizer_file, 'rb') as f:
        lb = pickle.load(f)
    tracker.stop_task()

    # Load and preprocess the image
    tracker.start_task('load and preprocess image')
    image = load_and_preprocess_image(image_path)
    tracker.stop_task()

    # Perform prediction
    tracker.start_task('perform prediction')
    predictions = model.predict(image)
    tracker.stop_task()

    tracker.start_task('process predictions')
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_confidences = predictions[0][top_5_indices]

    # Get the labels for the top 5 predictions
    top_5_labels = [lb.classes_[i] for i in top_5_indices]

    print('Top 5 Predictions:')
    for label, confidence in zip(top_5_labels, top_5_confidences):
        print(f'{label}: {confidence:.2f}')
    tracker.stop_task()

    tracker.stop()

if __name__ == '__main__':
    image_path = os.path.join(output_path, 'test.jpg')

    predict(image_path)
