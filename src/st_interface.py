import os
import random
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import streamlit as st
from PIL import Image

# Paths
test_images_folder = 'out/test_images'
output_path = 'out'
model_file_1 = os.path.join(output_path, 'grocerynet50_10epochs.h5')
model_file_2 = os.path.join(output_path, 'grocerynet50_30epochs.keras')
label_binarizer_file = os.path.join(output_path, 'label_binarizer.pkl')

# Load and preprocess image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Prediction function
def predict(image_path, model_path):
    model = load_model(model_path)
    with open(label_binarizer_file, 'rb') as f:
        lb = pickle.load(f)
    image = load_and_preprocess_image(image_path)
    predictions = model.predict(image)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    return [
        f"{i + 1}. {lb.classes_[idx]}: {predictions[0][idx] * 100:.2f}%"
        for i, idx in enumerate(top_5_indices)
    ]

# Get random images
def get_random_images(folder, num_images=20):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(tuple(image_extensions))
    ]
    return random.sample(image_files, min(num_images, len(image_files)))

# Initialize
if "evaluation_started" not in st.session_state:
    st.session_state.evaluation_started = False
    st.session_state.image_index = 0
    st.session_state.predictions = None

# Start evaluation
if not st.session_state.evaluation_started:
    user_id = st.text_input("Enter your User ID:")
    if st.button("Start Evaluation"):
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.image_index = 0
            st.session_state.predictions = None
            st.session_state.evaluation_started = True
            st.session_state.image_paths = get_random_images(test_images_folder)
        else:
            st.error("Please enter a valid User ID.")
else:
    # Show current image
    image_index = st.session_state.image_index
    if image_index >= len(st.session_state.image_paths):
        st.success("You have completed all evaluations!")
        st.session_state.evaluation_started = False
    else:
        image_path = st.session_state.image_paths[image_index]

        # Predict only once for the first image or after submitting feedback
        if st.session_state.predictions is None:
            st.session_state.predictions = {
                "model_1": predict(image_path, model_file_1),
                "model_2": predict(image_path, model_file_2),
            }

        # Display image and predictions side by side
        st.image(Image.open(image_path), caption=f"Image {image_index + 1}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model 1 Predictions")
            st.write("\n".join(st.session_state.predictions["model_1"]))
        with col2:
            st.subheader("Model 2 Predictions")
            st.write("\n".join(st.session_state.predictions["model_2"]))

        # Feedback form
        col1, col2 = st.columns(2)
        with col1:
            likert_1 = st.slider(
                "Model 1 Agreement (1-7)", 1, 7, 4, key=f"likert_1_{image_index}"
            )
            comments_1 = st.text_area("Comments for Model 1", key=f"comments_1_{image_index}")
        with col2:
            likert_2 = st.slider(
                "Model 2 Agreement (1-7)", 1, 7, 4, key=f"likert_2_{image_index}"
            )
            comments_2 = st.text_area("Comments for Model 2", key=f"comments_2_{image_index}")

        # Submit button
        if st.button("Submit"):
            # Save feedback
            feedback = pd.DataFrame({
                "User ID": [st.session_state.user_id],
                "Image": [image_path],
                "Model 1 Prediction": [", ".join(st.session_state.predictions["model_1"])],
                "Model 2 Prediction": [", ".join(st.session_state.predictions["model_2"])],
                "Likert Model 1": [likert_1],
                "Likert Model 2": [likert_2],
                "Comments Model 1": [comments_1],
                "Comments Model 2": [comments_2],
            })
            feedback.to_csv("feedback.csv", mode="a", header=not os.path.exists("feedback.csv"), index=False)

            # Move to the next image
            st.session_state.image_index += 1
            st.session_state.predictions = None
