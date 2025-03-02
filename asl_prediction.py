import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import requests
import time
from datetime import datetime


dataset_dir = "C:/Users/necha/hackhayward/dataSet/testingData"
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
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')) and image_count < 600:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = np.expand_dims(img, axis=-1) 
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
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
model.save('asl_model.h5')

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Set up webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define ROI size (larger to capture more of the hand)
roi_size = 300

# Preprocessing function for the ROI
def preprocess_roi(roi):
    """Preprocess ROI to match training data: grayscale, thresholding for white background with black hand outlines."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)      # Light blur to reduce noise
   
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2) 
    return thresh

# Groq API configuration
GROQ_API_KEY = "PLACE WITH YOUR API KEY" 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_groq_prediction(text):
    """Send text to Groq API and get a single word prediction."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192", 
        "messages": [
            {"role": "user", "content": f"Provide a single word that starts with: {text}"}
        ],
        "max_tokens": 10,  
        "temperature": 0.3 
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Groq Response: {response.json()}")  # Debug: Print the raw response
        predictions = response.json()['choices'][0]['message']['content'].strip()
        # Extract the first word (remove any extra text or punctuation)
        word = predictions.split()[0] if predictions else "No prediction"
        return word if word.isalpha() else "No prediction"  # Ensure it's a single alphabetic word
    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e.response.status_code if e.response else 'No response'} - {e.response.text if e.response else str(e)}")
        return f"Error: {str(e)}"
    except (KeyError, IndexError, TypeError) as e:
        print(f"Groq API Parsing Error: {str(e)} - Response: {response.json() if 'response' in locals() else 'No response'}")
        return "Error: Invalid response format"

# Variables for real-time letter tracking
current_letter = None
letter_start_time = None
letter_buffer = []  # List to store confirmed letters (e.g., ["H", "E"])
prediction_display = ""  # Store Groq’s word prediction

# Webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]
    print(f"Webcam resolution: {width}x{height}")  # Debug: check resolution

    # Define ROI coordinates (centered in the frame)
    start_x = (width - roi_size) // 2
    start_y = (height - roi_size) // 2
    end_x = start_x + roi_size
    end_y = start_y + roi_size

    # Ensure ROI fits within frame
    if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
        print("Warning: ROI size exceeds webcam resolution. Adjust roi_size.")
        break

    # Draw ROI rectangle on the frame (green outline)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Extract the ROI
    roi = frame[start_y:end_y, start_x:end_x]

    # Preprocess the ROI to get a binary image (white background, black hand)
    binary_roi = preprocess_roi(roi)

    # Resize to match model input size (64x64)
    resized_roi = cv2.resize(binary_roi, (IMG_SIZE, IMG_SIZE))

    # Normalize (binary image is already 0 or 255, so normalize to 0–1)
    normalized_roi = resized_roi.astype("float32") / 255.0

    # Add channel and batch dimensions
    input_roi = np.expand_dims(normalized_roi, axis=-1)  # Shape: (64, 64, 1)
    input_roi = np.expand_dims(input_roi, axis=0)        # Shape: (1, 64, 64, 1)

    # Predict the ASL letter
    prediction = model.predict(input_roi, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_letter = classes[predicted_class]

    # Track letter stability with a 3-second timer
    current_time = time.time()
    if current_letter == predicted_letter:
        if letter_start_time is None:
            letter_start_time = current_time
        elif current_time - letter_start_time >= 3.0:  # 3 seconds elapsed
            if predicted_letter not in letter_buffer:  # Avoid duplicates
                letter_buffer.append(predicted_letter)
                letter_start_time = None  # Reset timer after adding
                # Get Groq prediction for the current letter string
                letter_string = "".join(letter_buffer)
                prediction_display = get_groq_prediction(letter_string)
    else:
        current_letter = predicted_letter
        letter_start_time = current_time  # Reset timer for new letter

    # Display the prediction and instructions on the frame
    cv2.putText(frame, f'Predicted Letter: {predicted_letter}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Letter Buffer: { "".join(letter_buffer) }', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Groq Prediction: {prediction_display}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Place hand in larger green square", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize the processed ROI for display (larger window, e.g., 256x256)
    display_roi = cv2.resize(resized_roi, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Processed ROI", display_roi)  # Show larger version for visibility
    cv2.imshow("ASL Letter Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()