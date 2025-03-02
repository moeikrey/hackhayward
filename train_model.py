import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2  # Add this import statement
from utils import plot_training_history, visualize_predictions, preprocess_image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_HEIGHT, IMG_WIDTH = 28, 28  # Sign Language MNIST is 28x28
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 10  # Update to 10 classes for digits 0-9

def load_data():
    """Load Sign Language MNIST dataset from local directory"""
    
    print("Loading Sign Language MNIST dataset from local directory...")
    
    data_dir = "./sign_language_mnist/Sign-Language-Digits-Dataset-master/Dataset"
    
    # Initialize lists to hold images and labels
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    # Load images from each class folder
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Directory not found: {class_dir}")
        
        # Load images from the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = preprocess_image(cv2.imread(img_path), target_size=(IMG_HEIGHT, IMG_WIDTH))
            
            # Split into train and test sets (80% train, 20% test)
            if np.random.rand() < 0.8:
                train_images.append(img)
                train_labels.append(label)
                class_counts[label] += 1
            else:
                test_images.append(img)
                test_labels.append(label)
    
    # Print class counts
    print("Class counts:", class_counts)
    
    # Convert lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Add channel dimension to images
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    # Debugging information
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Train images range: {train_images.min()} - {train_images.max()}")
    print(f"Test images range: {test_images.min()} - {test_images.max()}")
    
    # Ensure there are enough samples
    if len(train_images) < BATCH_SIZE or len(test_images) < BATCH_SIZE:
        raise ValueError("Not enough samples in the dataset. Please ensure there are enough images in each class folder.")
    
    # One-hot encode labels
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    
    # Create augmented training data
    train_data_gen = datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE)
    
    # Create TensorFlow datasets
    ds_train = tf.data.Dataset.from_generator(
        lambda: train_data_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Create a validation split
    ds_val = ds_train.take(len(train_images) // 5)
    ds_train = ds_train.skip(len(train_images) // 5)
    
    # Create a mapping from index to letter
    idx_to_letter = {i: chr(48 + i) for i in range(NUM_CLASSES)}  # Map to digits 0-9
    
    print(f"Dataset loaded with {NUM_CLASSES} classes.")
    print(f"Index to letter mapping: {idx_to_letter}")
    
    return ds_train, ds_val, ds_test, idx_to_letter, test_images, np.argmax(test_labels, axis=1)

def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """Build the CNN model for Sign Language MNIST"""
    
    print("Building model...")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Reduce overfitting
        Dense(NUM_CLASSES, activation='softmax')  # Output layer
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, ds_train, ds_val, steps_per_epoch):
    """Train the model with early stopping and checkpoints"""
    
    print("Training model...")
    
    # Create directory for model checkpoints if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath='models/best_asl_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/final_asl_model.h5')
    print("Model saved to models/final_asl_model.h5")
    
    return history

def evaluate_model(model, ds_test):
    """Evaluate the model on test data"""
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    return test_accuracy, test_loss

def main():
    # Load data
    ds_train, ds_val, ds_test, idx_to_letter, test_images, test_labels = load_data()
    
    # Build model
    model = build_model()
    
    # Calculate steps per epoch
    steps_per_epoch = len(test_images) // BATCH_SIZE
    
    # Train model
    history = train_model(model, ds_train, ds_val, steps_per_epoch)
    
    # Evaluate model
    evaluate_model(model, ds_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize predictions
    visualize_predictions(model, test_images, test_labels, idx_to_letter)
    
    print("Training complete!")
    print("Run webcam_interpreter.py to test the model with your webcam.")

if __name__ == "__main__":
    main()