import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set the path to the input and output directories
in_path = 'in'
data_path = os.path.join(in_path, 'combined_dataset')
output_path = 'out'
test_images_path = os.path.join(output_path, 'test_images')

# Function to load data from images
def load_data_from_images(data_path):
    image_paths = []
    labels = []

    for master_folder in tqdm(sorted(os.listdir(data_path)), desc='Processing master folders...'):
        master_folder_path = os.path.join(data_path, master_folder)
        for subclass_folder in tqdm(sorted(os.listdir(master_folder_path)), desc=f'Processing master folder: {master_folder}'):
            subclass_folder_path = os.path.join(master_folder_path, subclass_folder)
            for root, _, files in os.walk(subclass_folder_path):
                for image_file in files:
                    if image_file.endswith('.jpg'):
                        image_path = os.path.join(root, image_file)
                        image_paths.append(image_path)
                        labels.append(subclass_folder)  # Use subclass folder as label

    return np.array(image_paths), np.array(labels)

# Function to generate a folder of test images
def generate_test_images(data_path, test_images_path):
    # Load data & labels from images
    image_paths, labels = load_data_from_images(data_path)

    # Split data into training and testing sets
    _, X_test, _, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Create the test images folder if it doesn't exist
    if not os.path.exists(test_images_path):
        os.makedirs(test_images_path)

    # Copy test images to the test images folder
    for i, (image_path, label) in enumerate(tqdm(zip(X_test, y_test), desc='Copying test images')):
        subclass_folder = label
        dest_path = os.path.join(test_images_path, f'{subclass_folder}_test_image_{i}.jpg')
        shutil.copy(image_path, dest_path)

    print(f'Test images saved to {test_images_path}')

if __name__ == '__main__':
    generate_test_images(data_path, test_images_path)
