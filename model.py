import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, callbacks, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report


# Function to load preprocessed data from .npy files
def load_preprocessed_data(save_dir):
    x_train = np.load(os.path.join(save_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(save_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    return x_train, x_test, y_train, y_test


# Preprocess the labels to one-hot encode them
def preprocess_labels(y_train, y_test, n_classes):
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    return y_train, y_test


# Create the CNN model with adjustments
def create_model(input_shape, n_classes):
    model = models.Sequential()
    model.add(Input(shape=input_shape))

    # Convolutional layers with Batch Normalization, Dropout, and L2 regularization
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Flatten and fully connected layers with Dropout
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(n_classes, activation='softmax'))

    # Compile the model with a reduced learning rate and gradient clipping
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Main function to train the model
if __name__ == "__main__":
    # Load preprocessed data
    save_dir = 'preprocessed_data'
    x_train, x_test, y_train, y_test = load_preprocessed_data(save_dir)

    # Determine the number of unique classes
    n_classes = len(np.unique(y_train))

    # One-hot encode the labels
    y_train, y_test = preprocess_labels(y_train, y_test, n_classes)

    # Define input shape and number of classes
    input_shape = (128, 128, 1)

    # Create the CNN model
    model = create_model(input_shape, n_classes)

    # Define callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test),
              callbacks=[checkpoint, early_stopping])

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}')

    # Generate predictions for the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:\n", cm)

    # Classification Report (Precision, Recall, F1-score)
    cr = classification_report(y_true, y_pred_classes)
    print("Classification Report:\n", cr)

    # Save the final trained model
    model.save('font_recognition_model.keras')
