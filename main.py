# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the dataset
def load_dataset():
    # Load the dataset of ASL hand sign images
    dataset = []
    for i in range(26):  # 26 letters of the alphabet
        img = cv2.imread(f"image_{i}.jpg")
        img = cv2.resize(img, (224, 224))  # Resize the image to 224x224 pixels
        dataset.append(img)
    return dataset

# Train the machine learning model
def train_model(dataset):
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    
    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Train the model
    model.fit(dataset, epochs=10, batch_size=32)
    return model

# Evaluate the trained model
def evaluate_model(model, test_dataset):
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {accuracy:.2f}")

# Integrate the trained model with a real-world application
def integrate_model(model):
    # Load a video stream from a webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame and pass it through the model
        frame = cv2.resize(frame, (224, 224))
        output = model.predict(frame)
        print(output)
        
        # Display the output
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    dataset = load_dataset()
    model = train_model(dataset)
    test_dataset = load_dataset()  # Load a separate test dataset
    evaluate_model(model, test_dataset)
    integrate_model(model)

if __name__ == "__main__":
    main()