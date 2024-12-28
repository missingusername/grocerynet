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
label_binarizer_file = os.path.join(output_path, 'label_binarizer.pkl')
label_binarizer = pickle.load(open(label_binarizer_file, 'rb'))

@st.cache_resource
def load_model_path(model_path, epoch_count):
    model_file_h5 = os.path.join(model_path, epoch_count, f'grocerynet50_{epoch_count}epochs.h5')
    model_file_keras = os.path.join(model_path, epoch_count, f'grocerynet50_{epoch_count}epochs.keras')
    if os.path.exists(model_file_h5):
        model_file = model_file_h5
    elif os.path.exists(model_file_keras):
        model_file = model_file_keras
    else:
        raise FileNotFoundError(f"No model file found for epoch {epoch_count} in {model_path}")
    model = load_model(model_file)
    return model

model_1 = load_model_path(output_path, '5')
model_2 = load_model_path(output_path, '30')

# Load and preprocess image
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Prediction function
def predict(image_path, model, label_binarizer):
    try:
        image = load_and_preprocess_image(image_path)
        
        predictions = model.predict(image)
        
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        return [
            f"{i + 1}. {label_binarizer.classes_[idx]}: {predictions[0][idx] * 100:.2f}%"
            for i, idx in enumerate(top_5_indices)
        ]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise

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
    st.session_state.image_paths = []
    for i in range(30):  # Assuming a maximum of 30 images
        st.session_state[f"predictions_model_1_{i}"] = []
        st.session_state[f"predictions_model_2_{i}"] = []
        st.session_state[f"likert_1_{i}"] = 4
        st.session_state[f"likert_2_{i}"] = 4
        st.session_state[f"comments_1_{i}"] = ""
        st.session_state[f"comments_2_{i}"] = ""

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
            if not st.session_state[f"predictions_model_1_{i}"]:
                st.session_state[f"predictions_model_1_{i}"] = predict(image_path, model_1, label_binarizer)
            st.write("\n".join(st.session_state[f"predictions_model_1_{i}"]))

        with col2:
            st.subheader("Model 2 Predictions")
            if not st.session_state[f"predictions_model_2_{i}"]:
                st.session_state[f"predictions_model_2_{i}"] = predict(image_path, model_2, label_binarizer)
            st.write("\n".join(st.session_state[f"predictions_model_2_{i}"]))

        # Feedback forms
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Model 1 Agreement (1-7)",
                1, 7, st.session_state[f"likert_1_{i}"],
                key=f"likert_1_{i}"
            )
            st.text_area(
                "Comments for Model 1",
                st.session_state[f"comments_1_{i}"],
                key=f"comments_1_{i}"
            )
        with col2:
            st.slider(
                "Model 2 Agreement (1-7)",
                1, 7, st.session_state[f"likert_2_{i}"],
                key=f"likert_2_{i}"
            )
            st.text_area(
                "Comments for Model 2",
                st.session_state[f"comments_2_{i}"],
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
