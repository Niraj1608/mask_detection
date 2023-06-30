import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import PIL.Image

model = load_model("mask_detection_model.model")

def main():
    st.title("Mask Detection App")
    st.write("Please choose an option")

    option = st.selectbox("Select an option", ("Upload Image", "Capture Photo"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read and preprocess the image
            image = PIL.Image.open(uploaded_file)
            image = image.resize((128, 128))  # Resize the image
            image = np.array(image)  # Convert to NumPy array
            image = image / 255.0  # Normalize the image
            input_image_reshaped = np.reshape(image, [1, 128, 128, 3])

            # Make predictions
            predictions = model.predict(input_image_reshaped)
            input_pred_label = np.argmax(predictions)

            # Display the result
            if input_pred_label == 1:
                st.write("Prediction: With Mask")
            else:
                st.write("Prediction: Without Mask")

            # Display the uploaded image
            st.write("")
            st.write("**Uploaded Image**")
            st.image(image, use_column_width=True)

    elif option == "Capture Photo":
        video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        video_capture.release()

        # Convert captured frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and preprocess the image
        image = cv2.resize(frame, (128, 128))
        image = image / 255.0
        input_image_reshaped = np.reshape(image, [1, 128, 128, 3])

        # Make predictions
        predictions = model.predict(input_image_reshaped)
        input_pred_label = np.argmax(predictions)

        # Display the result
        if input_pred_label == 1:
            st.write("Prediction: With Mask")
        else:
            st.write("Prediction: Without Mask")

        # Display the captured photo
        st.image(frame, channels="RGB", use_column_width=True)

if __name__ == '__main__':
    main()

