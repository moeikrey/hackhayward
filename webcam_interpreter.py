import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils import preprocess_image

def load_asl_model():
    """Load the trained ASL model"""
    model_path = 'models/best_asl_model.h5'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}!")
        print("Please run train_model.py first to train the model.")
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return model

def get_idx_to_digit():
    """Get mapping from index to digit"""
    idx_to_digit = {i: str(i) for i in range(10)}
    return idx_to_digit

def interpret_asl_webcam():
    """Run real-time ASL interpretation using webcam"""
    # Load model
    model = load_asl_model()
    if model is None:
        return
    
    # Get digit mapping
    idx_to_digit = get_idx_to_digit()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Variables for tracking predictions
    text = ""
    last_digit = None
    stable_counter = 0
    predictions = []
    
    print("ASL Interpreter started. Press 'q' to quit, 'c' to clear text.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Get region of interest (center square of frame)
        h, w = frame.shape[:2]
        square_size = min(h, w)
        offset_y = (h - square_size) // 2
        offset_x = (w - square_size) // 2
        
        # Create ROI box with some padding to focus on hand
        padding_ratio = 0.3  # Use 30% of the frame size
        roi_size = int(square_size * (1 - padding_ratio))
        roi_offset_y = offset_y + (square_size - roi_size) // 2
        roi_offset_x = offset_x + (square_size - roi_size) // 2
        
        # Extract ROI
        roi = frame[roi_offset_y:roi_offset_y + roi_size, 
                    roi_offset_x:roi_offset_x + roi_size]
        
        # Draw ROI rectangle
        cv2.rectangle(frame, 
                      (roi_offset_x, roi_offset_y), 
                      (roi_offset_x + roi_size, roi_offset_y + roi_size), 
                      (0, 255, 0), 2)
        
        # Preprocess image for model
        if roi.size > 0:  # Make sure ROI is not empty
            # Convert to grayscale, resize to 28x28, and normalize
            processed_roi = preprocess_image(roi, target_size=(28, 28))
            
            # Prepare for model input (add batch and channel dimensions)
            input_img = np.expand_dims(processed_roi, axis=0)
            input_img = np.expand_dims(input_img, axis=-1)
            
            # Make prediction
            prediction = model.predict(input_img, verbose=0)
            pred_class = np.argmax(prediction)
            confidence = prediction[0][pred_class]
            digit = idx_to_digit[pred_class]
            
            # Debugging information
            print(f"Prediction: {prediction}")
            print(f"Predicted class: {pred_class}, Digit: {digit}, Confidence: {confidence}")
            
            # Add prediction to list
            predictions.append(pred_class)
            if len(predictions) > 10:  # Keep only last 10 predictions
                predictions.pop(0)
            
            # Stabilize predictions to avoid flickering
            if len(predictions) >= 5:
                # Get most common prediction in last 5 frames
                most_common = max(set(predictions[-5:]), key=predictions[-5:].count)
                count = predictions[-5:].count(most_common)
                
                # Only accept if prediction is stable and confident
                if count >= 3 and confidence > 0.7:
                    # If new digit detected
                    if most_common != last_digit:
                        text += idx_to_digit[most_common]
                        last_digit = most_common
                        stable_counter = 0
                    else:
                        stable_counter += 1
                        # Reset after delay to allow repeated digits
                        if stable_counter > 20:
                            last_digit = None
            
            # Display prediction on frame
            cv2.putText(frame, f"{digit}: {confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display text at bottom of frame
        cv2.putText(frame, f"Text: {text[-30:]}", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Place hand sign in green box", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, "Press 'q' to quit, 'c' to clear", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Display frame
        cv2.imshow('ASL Interpreter', frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Clear text
            text = ""
            last_digit = None
            stable_counter = 0
            predictions = []
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    interpret_asl_webcam()