import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(image, target_size=(28, 28), grayscale=True):
    """
    Preprocess an image for model input
    
    Args:
        image: Input image
        target_size: Size to resize the image to
        grayscale: Whether to convert to grayscale
        
    Returns:
        Preprocessed image
    """
    if grayscale and len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize pixel values
    image = image / 255.0
    
    return image

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, test_images, true_labels, idx_to_letter, num_examples=10):
    """
    Visualize model predictions on test images
    
    Args:
        model: Trained model
        test_images: List of test images
        true_labels: List of true labels
        idx_to_letter: Dictionary mapping indices to letters
        num_examples: Number of examples to visualize
    """
    # Make predictions
    predictions = model.predict(test_images[:num_examples])
    pred_classes = np.argmax(predictions, axis=1)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        
        # Handle grayscale vs RGB images
        img = test_images[i]
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img)
            
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        
        # Get true and predicted labels
        true_label = idx_to_letter[true_labels[i]] if isinstance(true_labels[i], int) else idx_to_letter[np.argmax(true_labels[i])]
        pred_label = idx_to_letter[pred_classes[i]]
        
        # Display result
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()