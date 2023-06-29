import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog
import os

# Load the face mask detector model
model = load_model("mask_detection_model.model")

# Initialize the GUI
root = tk.Tk()
root.title("Mask Detection")
root.geometry("500x400")

# Function to classify the image
def classify_image(image):
    # Preprocess the image
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0)
    predictions = model.predict(expanded_image)
    if predictions[0][0] > 0.5:
        result = "Without Mask"
    else:
        result = "With Mask"

        # Update the result label text
    result_label.configure(text="Prediction: " + result)




# Function to handle file upload
def upload_image():
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File",
                                               filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    if file_path:
         # Display the uploaded image in the GUI
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        photo_label.configure(image=photo)
        photo_label.image = photo

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classify_image(image)




# Function to handle capturing photo from webcam
def capture_photo():
    video_capture = cv2.VideoCapture(0)
    _, frame = video_capture.read()
    video_capture.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    classify_image(frame)

    # Display the captured photo in the GUI
    image = Image.fromarray(frame)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)
    photo_label.configure(image=photo)
    photo_label.image = photo


result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)


# Create the GUI components
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

photo_label = tk.Label(root)
photo_label.pack()

# Run the GUI main loop
root.mainloop()
