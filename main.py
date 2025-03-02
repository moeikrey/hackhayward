import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset

# jc


def load_dataset():
    dataset = []
    labels = []
    for i in range(26):  # 26 letters of the alphabet
        for j in range(10):  # 10 images per letter
            img = cv2.imread(f"image_{i}_{j}.jpg")
            img = cv2.resize(img, (224, 224))  # Resize the image to 224x224 pixels
            dataset.append(img)
            labels.append(i)
    dataset = np.array(dataset)
    labels = to_categorical(labels)
    return dataset, labels

# Train the machine learning model


def train_model(dataset, labels):
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

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
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    return model

# Evaluate the trained model


def evaluate_model(model, x_test, y_test):
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)
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
        output = model.predict(np.array([frame]))
        predicted_class = np.argmax(output)
        print(f"Predicted class: {predicted_class}")

        # Display the output
        cv2.putText(frame, f"Predicted class: {predicted_class}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function


def main():
    dataset, labels = load_dataset()
    model = train_model(dataset, labels)
    evaluate_model(model, dataset[100:], labels[100:])
    integrate_model(model)


if __name__ == "__main__":
    main()
