import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('best_model.h5')
print("Model loaded successfully.")

# Class names
class_names = ['Ripe', 'Unripe', 'Overripe']  # Update based on your dataset

# Function to preprocess the uploaded image and make predictions
def classify_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Preprocess image
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    print(f"Predictions: {predictions}")  # Debug: Print the predictions
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]
    print(f"Predicted class index: {predicted_class}, label: {predicted_label}")  # Debug: Print predicted class info
    
    # Update the result label with the predicted class
    result_label.config(text=f"Predicted Class: {predicted_label}", font=("Arial", 18, "bold"))

# Function to open a file dialog and select an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        
        # Classify the uploaded image
        classify_image(file_path)

# Create the main application window
window = tk.Tk()
window.title("Palm Oil Fruit Classifier")
window.geometry("600x600")  # Set window size
window.configure(bg="#2C2F33")  # Set a darker background color

# Create a title label
title_label = Label(window, text="Palm Oil Fruit Classifier", font=("Arial", 24, "bold"), bg="#2C2F33", fg="#FFFFFF")
title_label.pack(pady=20)

# Create a frame for the upload button and image display
frame = Frame(window, bg="#2C2F33")
frame.pack(pady=20)

# Create a button to upload an image
upload_button = Button(frame, text="Upload Image", command=upload_image, font=("Arial", 16), bg="#7289DA", fg="white", padx=10, pady=5)
upload_button.pack(pady=10)

# Label to display the uploaded image
image_label = Label(frame, bg="#2C2F33", borderwidth=2, relief="groove")
image_label.pack(pady=10)

# Label to display the classification result
result_label = Label(window, text="Predicted Class: ", font=("Arial", 18), bg="#2C2F33", fg="#FFFFFF")
result_label.pack(pady=20)

# Start the Tkinter event loop
window.mainloop()
