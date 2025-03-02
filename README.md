# hackhayward 2025 3/1/25
Project by Andreas Sotiras, Jonathan Cornejo, Nechar KC, Luan Luc

This project is a hackathon project that aims to create a sign language interpreter trained off of images of the letters for ASL.

## Example Images
Below are some example images used for training the model. These images are located in the `examples` folder.

### Example 1
![Example 1](examples/A_example.png)

### Example 2
![Example 2](examples/B_example.png)

### Example 3
![Example 3](examples/C_example.png)

## Usage
This project uses a Convolutional Neural Network (CNN) to recognize ASL letters from webcam input. The model is trained on grayscale images of ASL letters and can predict the letter shown in the webcam feed.

### Steps to Use
1. **Install Dependencies**: Ensure you have all the required dependencies installed. You can install them using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

2. **Prepare Dataset**: Place your dataset of ASL letter images in the [testingData](http://_vscodecontentref_/1) directory. The images should be organized into subdirectories named after the corresponding letters.

3. **Train the Model**: Run the [main_2.py](http://_vscodecontentref_/2) script to train the model. The script will load the dataset, train the model, and save it as [asl_model.h5](http://_vscodecontentref_/3).

4. **Run the Webcam ASL Interpreter**: After training, the script will start the webcam and begin predicting ASL letters. If the predicted letter remains the same for 3 seconds, it will be confirmed and displayed on the screen.

    ```bash
    python main_2.py
    ```

5. **Interpreting ASL Letters**: Place your hand within the green square on the webcam feed. The model will predict the ASL letter and display it on the screen. If the letter remains the same for 3 seconds, it will be confirmed.

### Example Output
![Example Output](examples/output.png)

## Contributors
- Andreas Sotiras
- Jonathan Cornejo
- Nechar KC
- Luan Luc