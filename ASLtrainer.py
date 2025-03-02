# This will train you how to make the sign language hands.
# moeikrey 3/1/25
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import random
import time

# Dataset directory (update this path to your dataset)
dataset_dir = "./dataSet/testingData"
IMG_SIZE = 64  # Size expected by the model

# List of class labels (e.g., '0', 'A' to 'Z')
classes = sorted(os.listdir(dataset_dir))

# Load and prepare the dataset
data = []
labels = []

for label, cls in enumerate(classes):
    class_dir = os.path.join(dataset_dir, cls)
    if os.path.isdir(class_dir):
        print(f"Loading class: {cls}")
        image_count = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')) and image_count < 500:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    data.append(img)
                    labels.append(label)
                    image_count += 1
                else:
                    print(f"  Failed to load image: {img_path}")

# Convert to arrays and normalize
data = np.array(data, dtype="float32") / 255.0
labels = to_categorical(np.array(labels), num_classes=len(classes))

# Split into training and validation sets
if len(data) > 0:
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Loaded {len(data)} images across {len(classes)} classes.")
else:
    print("No data loaded. Please check your dataset directory.")
    exit()

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=32)  # Updated to 8 epochs
model.save('asl_model.h5')

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Set up webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define ROI size (smaller to focus tightly on the hand)
roi_size = 150

# Preprocessing function for the ROI
def preprocess_roi(roi):
    """Preprocess ROI to match training data: grayscale, thresholding for white background with black hand outlines."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)      # Light blur to reduce noise
    # Apply adaptive thresholding to create white background with black hand (no inversion)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)  # White background, black hand
    return thresh

# Function to display feedback
def display_feedback(frame, text, color):
    cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Select a random target letter
target_letter = random.choice(classes)
last_prediction = None
prediction_start_time = None
confirmation_time = 3  # seconds
score = 0  # Initialize score

# Load the sign chart image
sign_chart = cv2.imread('signchart.jpg')

# Webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Define ROI coordinates (centered in the frame)
    start_x = (width - roi_size) // 2
    start_y = (height - roi_size) // 2
    end_x = start_x + roi_size
    end_y = start_y + roi_size

    # Draw ROI rectangle on the frame (green outline)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Extract the ROI
    roi = frame[start_y:end_y, start_x:end_x]

    # Preprocess the ROI to get a binary image (white background, black hand)
    binary_roi = preprocess_roi(roi)

    # Resize to match model input size
    resized_roi = cv2.resize(binary_roi, (IMG_SIZE, IMG_SIZE))

    # Normalize (binary image is already 0 or 255, so normalize to 0–1)
    normalized_roi = resized_roi.astype("float32") / 255.0

    # Add channel and batch dimensions
    input_roi = np.expand_dims(normalized_roi, axis=-1)  # Shape: (64, 64, 1)
    input_roi = np.expand_dims(input_roi, axis=0)        # Shape: (1, 64, 64, 1)

    # Predict the ASL letter
    prediction = model.predict(input_roi, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = classes[predicted_class]

    # Display the target letter
    cv2.putText(frame, f'Target: {target_letter}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the prediction and instructions on the frame
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Place hand to fill green square completely", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Provide feedback based on the prediction
    current_time = time.time()
    if predicted_label == target_letter:
        if last_prediction == predicted_label:
            if current_time - prediction_start_time >= confirmation_time:
                display_feedback(frame, "Correct", (0, 255, 0))
                score += 1  # Increment score
                target_letter = random.choice(classes)  # Switch to the next letter
                last_prediction = None
                prediction_start_time = None
        else:
            last_prediction = predicted_label
            prediction_start_time = current_time
    else:
        display_feedback(frame, "Incorrect", (0, 0, 255))
        last_prediction = None
        prediction_start_time = None

    # Display the score
    cv2.putText(frame, f'Score: {score}', (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the processed ROI for debugging
    cv2.imshow("Processed ROI", resized_roi)
    cv2.imshow("ASL Letter Detection", frame)

    # Show the sign chart
    if sign_chart is not None:
        cv2.imshow("Sign Chart", sign_chart)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()