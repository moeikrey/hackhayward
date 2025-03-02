import tensorflow as tf
import numpy as np
import os
import cv2

# Path to dataset and model
dataset_dir = "./dataSet/testingData"
IMG_SIZE = 64  # Image size for processing
BATCH_SIZE = 32

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

tf.get_logger().setLevel('ERROR')

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get class labels
class_labels = list(train_data.class_indices.keys())
print("Class labels:", class_labels)

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=3, verbose=2)

# Save Model
model.save('asl_model.keras')

# Function to preprocess frame for prediction


def preprocess_frame(frame, img_size=64):
    roi_size = 150  # ROI size should match what you trained on
    height, width = frame.shape[:2]
    start_x = (width - roi_size) // 2
    start_y = (height - roi_size) // 2
    end_x = start_x + roi_size
    end_y = start_y + roi_size

    # Extract ROI
    roi = frame[start_y:end_y, start_x:end_x]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    gray = cv2.equalizeHist(gray)
    resized_roi = cv2.resize(gray, (img_size, img_size))
    normalized_roi = resized_roi.astype("float32") / 255.0
    input_roi = np.expand_dims(normalized_roi, axis=-1)
    input_roi = np.expand_dims(input_roi, axis=0)  # Shape: (1, 64, 64, 1)

    return input_roi

# Modified function to work with Streamlit


def start_camera():
    # Load the model and class labels
    try:
        loaded_model = tf.keras.models.load_model('asl_model.keras')
        loaded_class_labels = sorted(os.listdir(dataset_dir))
        print("Model and class labels loaded successfully")
    except Exception as e:
        print(f"Error loading model or class labels: {e}")
        loaded_model = model  # Use the model we just trained
        loaded_class_labels = class_labels

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Adjust brightness
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

            # Draw ROI rectangle
            roi_size = 150
            height, width = frame.shape[:2]
            start_x = (width - roi_size) // 2
            start_y = (height - roi_size) // 2
            end_x = start_x + roi_size
            end_y = start_y + roi_size
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            input_roi = preprocess_frame(frame)  # Get processed ROI for prediction

            # Predict the ASL letter
            prediction = loaded_model.predict(input_roi, verbose=0)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_label = loaded_class_labels[predicted_index]  # Convert index to label
            confidence = prediction[0][predicted_index] * 100

            confidence_threshold = 0.7  # 70% confidence
            if confidence > confidence_threshold * 100:
                cv2.putText(frame, f'Sign: {predicted_label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Sign: Uncertain', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # For Streamlit, we need to yield the frame
            yield frame

            # Check if we should stop (this won't be used in Streamlit but kept for standalone usage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()


# For standalone usage
if __name__ == "__main__":
    # Use OpenCV window display instead of yielding frames
    cap = cv2.VideoCapture(0)
    loaded_model = tf.keras.models.load_model('asl_model.keras')
    loaded_class_labels = sorted(os.listdir(dataset_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI rectangle
        roi_size = 150
        height, width = frame.shape[:2]
        start_x = (width - roi_size) // 2
        start_y = (height - roi_size) // 2
        end_x = start_x + roi_size
        end_y = start_y + roi_size
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        input_roi = preprocess_frame(frame)
        prediction = loaded_model.predict(input_roi, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = loaded_class_labels[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        cv2.putText(frame, f'Sign: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# import tensorflow as tf
# import numpy as np
# import cv2
# import streamlit as st

# # Load the trained model
# model = tf.keras.models.load_model('asl_model.h5')

# # Set up webcam


# def start_camera():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Could not open webcam.")
#         return

#     roi_size = 150  # Region of Interest size
#     IMG_SIZE = 64

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Error: Failed to capture frame.")
#             break

#         # Get frame dimensions
#         height, width = frame.shape[:2]
#         start_x = (width - roi_size) // 2
#         start_y = (height - roi_size) // 2
#         end_x = start_x + roi_size
#         end_y = start_y + roi_size

#         # Extract and preprocess ROI
#         roi = frame[start_y:end_y, start_x:end_x]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         binary_roi = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                            cv2.THRESH_BINARY, 11, 2)

#         resized_roi = cv2.resize(binary_roi, (IMG_SIZE, IMG_SIZE))
#         normalized_roi = resized_roi.astype("float32") / 255.0
#         input_roi = np.expand_dims(normalized_roi, axis=-1)
#         input_roi = np.expand_dims(input_roi, axis=0)

#         # Predict the ASL letter
#         prediction = model.predict(input_roi, verbose=0)
#         predicted_class = np.argmax(prediction, axis=1)[0]

#         # Display prediction on frame
#         cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Convert frame to RGB for Streamlit
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Yield frame for Streamlit to display
#         yield frame

#     cap.release()
#     cv2.destroyAllWindows()


# ----- end-----

#  import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# import os
# import cv2
# from sklearn.model_selection import train_test_split

# # Dataset directory (update this path to your dataset)
# dataset_dir = "./dataSet/testingData"
# IMG_SIZE = 64  # Size expected by the model

# # List of class labels (e.g., '0', 'A' to 'Z')
# classes = sorted(os.listdir(dataset_dir))

# # Load and prepare the dataset
# data = []
# labels = []

# for label, cls in enumerate(classes):
#     class_dir = os.path.join(dataset_dir, cls)
#     if os.path.isdir(class_dir):
#         print(f"Loading class: {cls}")
#         image_count = 0
#         for img_name in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_name)
#             if img_name.lower().endswith(('png', 'jpg', 'jpeg')) and image_count < 500:
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#             # Check if file is an image and limit to 50 images
#             if img_name.lower().endswith(('png', 'jpg', 'jpeg')) and image_count < 50:
#                 img = cv2.imread(img_path)

#                 # Check if the image was successfully loaded
#                 if img is not None:
#                     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#                     img = np.expand_dims(img, axis=-1)  # Add channel dimension
#                     data.append(img)
#                     labels.append(label)
#                     image_count += 1
#                 else:
#                     print(f"  Failed to load image: {img_path}")

# # Convert to arrays and normalize
# data = np.array(data, dtype="float32") / 255.0
# labels = to_categorical(np.array(labels), num_classes=len(classes))

# # Split into training and validation sets
# if len(data) > 0:
#     X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
#     print(f"Loaded {len(data)} images across {len(classes)} classes.")
# else:
#     print("No data loaded. Please check your dataset directory.")
#     exit()

# # Define the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(classes), activation='softmax')
# ])

# # Compile and train the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=32)  # Updated to 8 epochs
# model.save('asl_model.h5')

# # Load the trained model
# model = tf.keras.models.load_model('asl_model.h5')

# # Set up webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Define ROI size (smaller to focus tightly on the hand)
# roi_size = 150

# # Preprocessing function for the ROI


# def preprocess_roi(roi):
#     """Preprocess ROI to match training data: grayscale, thresholding for white background with black hand outlines."""
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)      # Light blur to reduce noise
#     # Apply adaptive thresholding to create white background with black hand (no inversion)
#     thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)  # White background, black hand
#     return thresh


# # Webcam loop
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Get frame dimensions
#     height, width = frame.shape[:2]

#     # Define ROI coordinates (centered in the frame)
#     start_x = (width - roi_size) // 2
#     start_y = (height - roi_size) // 2
#     end_x = start_x + roi_size
#     end_y = start_y + roi_size

#     # Draw ROI rectangle on the frame (green outline)
#     cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

#     # Extract the ROI
#     roi = frame[start_y:end_y, start_x:end_x]

#     # Preprocess the ROI to get a binary image (white background, black hand)
#     binary_roi = preprocess_roi(roi)

#     # Resize to match model input size
#     resized_roi = cv2.resize(binary_roi, (IMG_SIZE, IMG_SIZE))

#     # Normalize (binary image is already 0 or 255, so normalize to 0–1)
#     normalized_roi = resized_roi.astype("float32") / 255.0

#     # Add channel and batch dimensions
#     input_roi = np.expand_dims(normalized_roi, axis=-1)  # Shape: (64, 64, 1)
#     input_roi = np.expand_dims(input_roi, axis=0)        # Shape: (1, 64, 64, 1)

#     # Predict the ASL letter
#     prediction = model.predict(input_roi, verbose=0)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     predicted_label = classes[predicted_class]

#     # Display the prediction and instructions on the frame
#     cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, "Place hand to fill green square completely", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Show the processed ROI for debugging
#     cv2.imshow("Processed ROI", resized_roi)
#     # Preprocess the frame (resize to IMG_SIZE and normalize)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = img / 255.0  # Normalize pixel values

#     # Predict the class of the image
#     prediction = model.predict(img)
#     predicted_class = np.argmax(prediction, axis=1)
#     predicted_label = classes[predicted_class[0]]  # Convert to class label

#     # Display the predicted letter on the frame
#     cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("ASL Letter Detection", frame)

#     # # Exit on 'q' key press
#     # if cv2.waitKey(10) & 0xFF == ord('q'):

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
