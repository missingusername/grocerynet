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
model_file_1 = os.path.join(output_path, 'grocerynet50_5epochs.keras')
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
def get_random_images(folder, num_images=30):  # Load 30 images for both models
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(tuple(image_extensions))
    ]
    return random.sample(image_files, min(num_images, len(image_files)))

# Initialize session state
if "evaluation_started" not in st.session_state:
    st.session_state.evaluation_started = False
    st.session_state.feedback = {}

# Start evaluation
user_id = st.text_input("Enter your User ID:")
if st.button("Start Evaluation") and user_id:
    st.session_state.evaluation_started = True
    st.session_state.image_paths = get_random_images(test_images_folder)

if st.session_state.evaluation_started:
    images_per_model = 15
    st.header("Evaluation")
    for i, image_path in enumerate(st.session_state.image_paths[:images_per_model]):
        st.subheader(f"Image {i + 1}")
        
        # Show image
        st.image(Image.open(image_path), caption=f"Image {i + 1}")

        # Predictions side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model 1 Predictions")
            if f"predictions_model_1_{i}" not in st.session_state:
                st.session_state[f"predictions_model_1_{i}"] = predict(image_path, model_file_1)
            st.write("\n".join(st.session_state[f"predictions_model_1_{i}"]))

        with col2:
            st.subheader("Model 2 Predictions")
            if f"predictions_model_2_{i}" not in st.session_state:
                st.session_state[f"predictions_model_2_{i}"] = predict(image_path, model_file_2)
            st.write("\n".join(st.session_state[f"predictions_model_2_{i}"]))

        # Feedback forms
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Model 1 Agreement (1-7)",
                1, 7, 4,
                key=f"likert_1_{i}"
            )
            st.text_area(
                "Comments for Model 1",
                key=f"comments_1_{i}"
            )
        with col2:
            st.slider(
                "Model 2 Agreement (1-7)",
                1, 7, 4,
                key=f"likert_2_{i}"
            )
            st.text_area(
                "Comments for Model 2",
                key=f"comments_2_{i}"
            )

    # Save feedback button
    if st.button("Submit All Feedback"):
        feedback = []
        for i, image_path in enumerate(st.session_state.image_paths[:images_per_model]):
            feedback.append({
                "User ID": user_id,
                "Image": image_path,
                "Model 1 Prediction": ", ".join(st.session_state[f"predictions_model_1_{i}"]),
                "Model 2 Prediction": ", ".join(st.session_state[f"predictions_model_2_{i}"]),
                "Likert Model 1": st.session_state[f"likert_1_{i}"],
                "Likert Model 2": st.session_state[f"likert_2_{i}"],
                "Comments Model 1": st.session_state[f"comments_1_{i}"],
                "Comments Model 2": st.session_state[f"comments_2_{i}"],
            })

        feedback_df = pd.DataFrame(feedback)
        feedback_df.to_csv("feedback.csv", mode="a", header=not os.path.exists("feedback.csv"), index=False)
        st.success("Feedback saved successfully!")
