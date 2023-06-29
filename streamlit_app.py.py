import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
from PIL import Image

model = load_model("mask_detection_model.model")

def main():
    st.title("Mask Detection App")
    st.write("Please choose an option")

    option = st.selectbox("Select an option", ("Upload Image", "Capture Photo"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read and preprocess the image
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (128, 128))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Make predictions
            predictions = model.predict(image)
            without_mask_prob = predictions[0][0]

            # Display the result
            if without_mask_prob > 0.5:
                st.write("Prediction: Without Mask")
            else:
                st.write("Prediction: With Mask")

            # Display the uploaded image
                # Display the uploaded image
                st.write("")
                st.write("**Uploaded Image**")
                st.image(image, use_column_width=True, caption='')

    elif option == "Capture Photo":
        video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        video_capture.release()

        # Convert captured frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and preprocess the image
        image = cv2.resize(frame, (128, 128))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Make predictions
        predictions = model.predict(image)
        without_mask_prob = predictions[0][0]

        # Display the result
        if without_mask_prob > 0.5:
            st.write("Prediction: Without Mask")
        else:
            st.write("Prediction: With Mask")

        # Display the captured photo
        st.image(frame, caption='Captured Photo')

if __name__ == '__main__':
    main()
