# Mask Detection Project

This project aims to detect whether a person is wearing a mask or not using deep learning techniques. It includes a Convolutional Neural Network (CNN) model trained on a dataset of masked and unmasked faces.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictive System with GUI](#predictive-system-with-gui)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Niraj1608/email-spam-classifier.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Change to the project directory:
    ```
    cd mask-detection
    ```


## Usage

To use the Mask Detection system, follow the instructions below.

1. Prepare your dataset of masked and unmasked face images.

2. Train the CNN model using the provided training script. See [Model Training](#model-training) section for details.

3. Evaluate the trained model using the provided evaluation script. See [Model Evaluation](#model-evaluation) section for details.

4. Use the trained model in a predictive system with GUI. See [Predictive System with GUI](#predictive-system-with-gui) section for details.

5. Alternatively, deploy the model using Streamlit and create an interactive web application. See [Streamlit App](#streamlit-app) section for details.

## Model Training

To train the CNN model, follow these steps:

1. Prepare your dataset of masked and unmasked face images. Make sure the images are organized in separate directories for each class.

2. Use the `sklearn.model_selection.train_test_split` function to split the dataset into training and validation sets. Adjust the parameters as needed.

3. Preprocess the images by resizing, normalizing, and converting them into the appropriate format for the CNN model.

4. Build the CNN model using TensorFlow and define the layers, activation functions, and optimizer.

5. Train the model using the training set and validate it using the validation set. Adjust the hyperparameters such as batch size, number of epochs, and learning rate as needed.

6. Save the trained model for future use.

## Model Evaluation

To evaluate the trained model, follow these steps:

1. Load the saved model using the `tensorflow.keras.models.load_model` function.

2. Prepare a separate test set or use a portion of the original dataset for evaluation.

3. Preprocess the test images in the same way as during training.

4. Use the loaded model to predict the labels for the test images.

5. Evaluate the model's performance by calculating metrics such as accuracy, precision, recall, and F1 score.

## Predictive System with GUI

The predictive system with GUI allows you to interactively test the trained model using a graphical user interface.

1. Run the `predictive_system_gui.py` script.

2. Upload an image of a person's face using the provided interface.

3. The system will process the image and predict whether the person is wearing a mask or not.

4. The result will be displayed on the GUI along with the uploaded image.

## Streamlit App

To run the Mask Detection app using Streamlit, follow these steps:

1. Run the `streamlit_app.py` script.

## License

This project does not have a specific license. All rights reserved.


## Acknowledgements
This project is built upon the contributions and resources from the open-source community, including scikit-learn, NLTK, and Matplotlib.

## contact
If you have any questions or inquiries, please contact nirajprmr1608@example.com.


