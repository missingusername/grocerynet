import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def get_next_file_number(dest_dir, base_name):
    existing_files = [f for f in os.listdir(dest_dir) if f.startswith(base_name)]
    if not existing_files:
        return 1
    existing_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    return max(existing_numbers) + 1

def combine_images(src_folders, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    total_files = sum([len(files) for src_folder in src_folders for r, d, files in os.walk(src_folder) if any(file.endswith(('.png', '.jpg', '.jpeg')) for file in files)])
    with tqdm(total=total_files, desc=f"Copying images to {dest_folder}") as pbar:
        for src_folder in src_folders:
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        relative_path = os.path.relpath(root, src_folder)
                        dest_dir = os.path.join(dest_folder, relative_path)
                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                        
                        base_name = '_'.join(file.split('_')[:-1])
                        next_number = get_next_file_number(dest_dir, base_name)
                        new_file_name = f"{base_name}_{next_number:03d}.jpg"
                        
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(dest_dir, new_file_name)
                        shutil.copy(src_file, dest_file)
                        pbar.update(1)

def create_hdf5(dataset_dir, hdf5_path):
    image_paths = []
    labels = []
    label_map = {}
    label_counter = 0

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                label = os.path.relpath(root, dataset_dir)
                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1
                labels.append(label_map[label])

def main():
    data_path = os.path.join('in', 'GroceryStoreDataset', 'dataset')
    categories = ['train', 'test', 'val']
    combined_dir = os.path.join('in', 'combined_dataset')

    src_folders = [os.path.join(data_path, category) for category in categories]
    combine_images(src_folders, combined_dir)

if __name__ == "__main__":
    main()
