import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Dataset directory
dataset_dir = "C:/Users/necha/Desktop/HACKATHON/a/dataSet/trainingData"  # Update the dataset path
IMG_SIZE = 64  # Image size to which we resize the images

# List of class labels (0, A-Z)
classes = sorted(os.listdir(dataset_dir))

# Prepare the data
data = []
labels = []

# Load and preprocess the dataset
for label, cls in enumerate(classes):
    class_dir = os.path.join(dataset_dir, cls)
    
    # Check if it's a directory
    if os.path.isdir(class_dir):
        print(f"Loading class: {cls} from {class_dir}")
        
        # Initialize a counter to limit the number of images
        image_count = 0
        
        # Iterate through images directly in the class folder
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Check if file is an image and limit to 50 images
            if img_name.lower().endswith(('png', 'jpg', 'jpeg')) and image_count < 50:
                img = cv2.imread(img_path)
                
                # Check if the image was successfully loaded
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize the image
                    data.append(img)
                    labels.append(label)
                    image_count += 1  # Increment the image counter
                else:
                    print(f"  Failed to load image: {img_path}")
            else:
                print(f"  Skipping non-image file: {img_name}")

# Convert data to numpy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values
labels = np.array(labels)

# One-hot encode the labels
labels = to_categorical(labels, num_classes=len(classes))

# Split the data into training and validation sets
if len(data) > 0:  # Avoid error if no data is loaded
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Loaded {len(data)} images and {len(labels)} labels.")
else:
    print("No data loaded. Check dataset directory and structure.")

# Define a CNN model for ASL letter recognition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('asl_model.h5')

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

# Set up the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame (resize to IMG_SIZE and normalize)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    
    # Predict the class of the image
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = classes[predicted_class[0]]  # Convert to class label
    
    # Display the predicted letter on the frame
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("ASL Letter Detection", frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
