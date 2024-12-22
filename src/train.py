import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from codecarbon import EmissionsTracker

# Define the hierarchical ResNet model using Keras
def create_hierarchical_resnet(num_master_classes, num_sub_classes, num_specific_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add additional dense layers with batch normalization and dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    master_out = layers.Dense(num_master_classes, activation='softmax', name='master_output')(x)
    sub_out = layers.Dense(num_sub_classes, activation='softmax', name='sub_output')(x)
    specific_out = layers.Dense(num_specific_classes, activation='softmax', name='specific_output')(x)

    model = models.Model(inputs, [master_out, sub_out, specific_out])
    return model

# Function to create hierarchical labels
def create_hierarchical_labels(dataset_dir, label_file='labels.json'):
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels['master_labels'], labels['sub_labels'], labels['specific_labels'], labels['master_to_idx'], labels['sub_to_idx'], labels['specific_to_idx']
    
    master_classes = sorted(os.listdir(dataset_dir))
    sub_classes = []
    specific_classes = []
    
    master_to_idx = {cls: idx for idx, cls in enumerate(master_classes)}
    sub_to_idx = {}
    specific_to_idx = {}
    
    master_labels = []
    sub_labels = []
    specific_labels = []
    
    for master_class in master_classes:
        print(f"Processing {master_class}")
        master_path = os.path.join(dataset_dir, master_class)
        sub_classes = sorted(os.listdir(master_path))
        
        for sub_class in sub_classes:
            print(f"Processing {sub_class}")
            sub_path = os.path.join(master_path, sub_class)
            specific_classes = sorted(os.listdir(sub_path))
            
            # Check if the sub_class contains images directly
            if all(os.path.isfile(os.path.join(sub_path, item)) for item in specific_classes):
                if sub_class not in sub_to_idx:
                    sub_to_idx[sub_class] = len(sub_to_idx)
                
                for img_name in specific_classes:
                    master_labels.append(master_to_idx[master_class])
                    sub_labels.append(sub_to_idx[sub_class])
                    specific_labels.append(sub_to_idx[sub_class])  # Use sub_class as specific label
            else:
                for specific_class in specific_classes:
                    print(f"Processing {specific_class}")
                    specific_path = os.path.join(sub_path, specific_class)
                    for img_name in os.listdir(specific_path):
                        if sub_class not in sub_to_idx:
                            sub_to_idx[sub_class] = len(sub_to_idx)
                        if specific_class not in specific_to_idx:
                            specific_to_idx[specific_class] = len(specific_to_idx)
                        
                        master_labels.append(master_to_idx[master_class])
                        sub_labels.append(sub_to_idx[sub_class])
                        specific_labels.append(specific_to_idx[specific_class])
    
    labels = {
        'master_labels': master_labels,
        'sub_labels': sub_labels,
        'specific_labels': specific_labels,
        'master_to_idx': master_to_idx,
        'sub_to_idx': sub_to_idx,
        'specific_to_idx': specific_to_idx
    }
    
    with open(label_file, 'w') as f:
        json.dump(labels, f)
    
    return master_labels, sub_labels, specific_labels, master_to_idx, sub_to_idx, specific_to_idx

# Custom data generator to include hierarchical labels
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_generator, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes):
        self.image_generator = image_generator
        self.master_labels = master_labels
        self.sub_labels = sub_labels
        self.specific_labels = specific_labels
        self.num_master_classes = num_master_classes
        self.num_sub_classes = num_sub_classes
        self.num_specific_classes = num_specific_classes
    
    def __len__(self):
        return len(self.image_generator)
    
    def __getitem__(self, index):
        images, _ = self.image_generator[index]
        batch_size = images.shape[0]
        master_labels = np.array(self.master_labels[index * batch_size:(index + 1) * batch_size])
        sub_labels = np.array(self.sub_labels[index * batch_size:(index + 1) * batch_size])
        specific_labels = np.array(self.specific_labels[index * batch_size:(index + 1) * batch_size])
        
        master_labels = tf.keras.utils.to_categorical(master_labels, num_classes=self.num_master_classes)
        sub_labels = tf.keras.utils.to_categorical(sub_labels, num_classes=self.num_sub_classes)
        specific_labels = tf.keras.utils.to_categorical(specific_labels, num_classes=self.num_specific_classes)
        
        return images, {'master_output': master_labels, 'sub_output': sub_labels, 'specific_output': specific_labels}

# Load dataset and create data generators with data augmentation
def load_dataset(dataset_dir, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes, batch_size=32, train_size=0.7, val_size=0.15, test_size=0.15):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_size + test_size,
        horizontal_flip=True,
        rotation_range=20,
        fill_mode='nearest',
        brightness_range=[0.9, 1.1]
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_test_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    val_size = int(val_size / (val_size + test_size) * val_test_generator.samples)
    val_generator = val_test_generator
    test_generator = val_test_generator

    train_data_gen = CustomDataGenerator(train_generator, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes)
    val_data_gen = CustomDataGenerator(val_generator, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes)
    test_data_gen = CustomDataGenerator(test_generator, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes)

    return train_data_gen, val_data_gen, test_data_gen

# Function to plot the training and validation loss
def plot_losses(history, output_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['master_output_accuracy'], label='Train Master Accuracy')
    plt.plot(history.history['val_master_output_accuracy'], label='Val Master Accuracy', linestyle=':')
    plt.plot(history.history['sub_output_accuracy'], label='Train Sub Accuracy')
    plt.plot(history.history['val_sub_output_accuracy'], label='Val Sub Accuracy', linestyle=':')
    plt.plot(history.history['specific_output_accuracy'], label='Train Specific Accuracy')
    plt.plot(history.history['val_specific_output_accuracy'], label='Val Specific Accuracy', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.show()

# Function to save the model
def save_model(model, output_dir):
    print("Saving model")
    model_path = os.path.join(output_dir, 'model.h5')
    model.save(model_path)

# Function to save the performance report
def save_performance_report(history, output_dir):
    report = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_master_accuracy': history.history['master_output_accuracy'],
        'val_master_accuracy': history.history['val_master_output_accuracy'],
        'train_sub_accuracy': history.history['sub_output_accuracy'],
        'val_sub_accuracy': history.history['val_sub_output_accuracy'],
        'train_specific_accuracy': history.history['specific_output_accuracy'],
        'val_specific_accuracy': history.history['val_specific_output_accuracy']
    }
    report_path = os.path.join(output_dir, 'performance_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f)
    print("Performance report saved")
    print(report)

# Function to evaluate the model on the test set and generate classification report
def evaluate_model(model, test_generator, output_dir):
    results = model.evaluate(test_generator)
    print("Test Loss and Accuracy:", results)

    # Generate predictions
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes

    # Convert predictions and true labels to categorical
    y_pred_master = np.argmax(y_pred[0], axis=1)
    y_pred_sub = np.argmax(y_pred[1], axis=1)
    y_pred_specific = np.argmax(y_pred[2], axis=1)

    # Generate classification report
    report_master = classification_report(y_true, y_pred_master, target_names=test_generator.class_indices.keys())
    report_sub = classification_report(y_true, y_pred_sub, target_names=test_generator.class_indices.keys())
    report_specific = classification_report(y_true, y_pred_specific, target_names=test_generator.class_indices.keys())

    # Save classification report
    with open(os.path.join(output_dir, 'classification_report_master.txt'), 'w') as f:
        f.write(report_master)
    with open(os.path.join(output_dir, 'classification_report_sub.txt'), 'w') as f:
        f.write(report_sub)
    with open(os.path.join(output_dir, 'classification_report_specific.txt'), 'w') as f:
        f.write(report_specific)

def main():
    output_dir = 'out'
    emissions_path = os.path.join(output_dir, 'emissions')
    if not os.path.exists(emissions_path):
        os.makedirs(emissions_path)

    tracker = EmissionsTracker(
        project_name="Hierarchical ResNet Training",
        experiment_id="Training",
        output_dir=emissions_path,
        output_file="emissions.csv"
    )

    tracker.start()

    tracker.start_task('setup directories')
    dataset_dir = os.path.join('in','combined_dataset')
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tracker.stop_task()

    tracker.start_task('label generation/loading')
    master_labels, sub_labels, specific_labels, master_to_idx, sub_to_idx, specific_to_idx = create_hierarchical_labels(dataset_dir, label_file='out/labels.json')
    tracker.stop_task()
    
    num_master_classes = len(master_to_idx)
    num_sub_classes = len(sub_to_idx)
    num_specific_classes = len(specific_to_idx)
    
    tracker.start_task('model setup')
    model = create_hierarchical_resnet(num_master_classes, num_sub_classes, num_specific_classes)
    tracker.stop_task()
    
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss={'master_output': 'categorical_crossentropy', 'sub_output': 'categorical_crossentropy', 'specific_output': 'categorical_crossentropy'},
                  metrics={'master_output': 'accuracy', 'sub_output': 'accuracy', 'specific_output': 'accuracy'})
    
    tracker.start_task('dataset loading')
    train_generator, val_generator, test_generator = load_dataset(dataset_dir, master_labels, sub_labels, specific_labels, num_master_classes, num_sub_classes, num_specific_classes, batch_size)
    tracker.stop_task()
    
    tracker.start_task('model training')
    history = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)
    tracker.stop_task()
    
    tracker.start_task('saving model and generating reports')
    plot_losses(history, output_dir)
    save_model(model, output_dir)
    save_performance_report(history, output_dir)
    tracker.stop_task()
    
    tracker.start_task('model evaluation')
    evaluate_model(model, test_generator, output_dir)
    tracker.stop_task()

    tracker.stop()

if __name__ == "__main__":
    main()
