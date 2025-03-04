import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================================
# Configuration
# ======================================

IMAGE_SIZE = (128, 128)  # Resize all images to this size
DATASET_DIR = 'generated_font_dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train', 'combinations')
TEST_DIR = os.path.join(DATASET_DIR, 'test', 'combinations')


# ======================================
# Helper Functions
# ======================================

def load_images_from_folder(folder_path):
    """
    Load images and labels from a given folder path.
    """
    images = []
    labels = []

    for font_folder in os.listdir(folder_path):
        font_folder_path = os.path.join(folder_path, font_folder)

        if not os.path.isdir(font_folder_path):
            continue

        for img_file in os.listdir(font_folder_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(font_folder_path, img_file)
                try:
                    # Load the image and resize it
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(IMAGE_SIZE)

                    # Convert the image to NumPy array
                    img_array = np.array(img)

                    images.append(img_array)
                    labels.append(font_folder)  # Use folder name as the label (font name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)


def preprocess_images(images):
    """
    Preprocess the images: normalize and reshape.
    """
    images = images.astype('float32') / 255.0  # Normalize pixel values
    images = np.expand_dims(images, axis=-1)  # Add a channel dimension (grayscale)
    return images


def encode_labels(labels):
    """
    Encode the labels (font names) into numerical format.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder.classes_


# ======================================
# Main Preprocessing Logic
# ======================================

def preprocess_dataset():
    """
    Load, preprocess, and split the dataset.
    """
    print("Loading training data...")
    train_images, train_labels = load_images_from_folder(TRAIN_DIR)

    print("Loading testing data...")
    test_images, test_labels = load_images_from_folder(TEST_DIR)

    print("Preprocessing images...")
    x_train = preprocess_images(train_images)
    x_test = preprocess_images(test_images)

    print("Encoding labels...")
    y_train, train_classes = encode_labels(train_labels)
    y_test, _ = encode_labels(test_labels)

    print(f"Train dataset shape: {x_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Test dataset shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    return x_train, x_test, y_train, y_test, train_classes


# ======================================
# Execution
# ======================================

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, classes = preprocess_dataset()

    print(f"Preprocessing complete. Number of classes: {len(classes)}")

    # Example: Saving the preprocessed arrays for further use (optional)
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('classes.npy', classes)
