import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load the saved model
model = tf.keras.models.load_model('best_model.h5')
print("Model loaded successfully.")
focal_length = 800  # Replace with your camera's focal length in pixels
baseline = 0.06 

# Class names
class_names = ['Ripe', 'Unripe', 'Overripe', 'Non-Palm']  # Update based on your dataset, 'Non-Palm' added for non-palm objects

# Function to detect fruit, draw bounding boxes, and estimate weight and price
def detect_fruit_and_draw_bounding_boxes(image_path, size_to_weight_ratio=0.0001):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image. Check the path.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur and Thresholding for better contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes and calculate weight and price only for palm oil fruit
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # If the detected object is not a palm fruit, skip weight and price estimation
        if predicted_label == 'Non-Palm':
            result_label1.config(text="Predicted Weight: N/A", font=("Arial", 18, "bold"))
            result_label4.config(text="Predicted Price (MYR): N/A", font=("Arial", 18, "bold"))
        else:
            estimated_weight = area * size_to_weight_ratio
            estimated_price = estimated_weight * 55.52

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Weight: {estimated_weight:.2f} kg", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            result_label1.config(text=f"Predicted Weight: {estimated_weight:.2f} kg", font=("Arial", 18, "bold"))
            result_label4.config(text=f"Predicted Price (MYR): {estimated_price:.2f}", font=("Arial", 18, "bold"))

def compute_distance(focal_length, baseline, disparity_value):
    if disparity_value > 0:
        return (focal_length * baseline) / disparity_value
    else:
        return None

def compute_disparity_and_distance(left_image_path, right_image_path):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        raise Exception("Could not load one or both images. Check the paths.")

    left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
    right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 9,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=32
    )

    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    disparity_map = cv2.medianBlur(disparity_map, 5)

    center = (disparity_map.shape[1] // 2, disparity_map.shape[0] // 2)
    disparity_value = disparity_map[center[1], center[0]]
    distance = compute_distance(focal_length, baseline, disparity_value)

    result_label2.config(text=f"Predicted Distance: {distance:.2f} m" if distance else "Unable to estimate distance", font=("Arial", 18, "bold"))
    
    if distance:
        print(f"Estimated distance to object at center: {distance:.2f} meters")
    else:
        print("Invalid disparity; unable to estimate distance.")

# Function to preprocess the uploaded image and make predictions
def classify_image(img_path):
    global predicted_label
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]
    
    result_label.config(text=f"Predicted Class: {predicted_label}", font=("Arial", 18, "bold"))

# Function to open a file dialog and select an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
  #  rfile_path = filedialog.askopenfilename(title="Select rightside Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
 #   lfile_path = filedialog.askopenfilename(title="Select leftside Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        classify_image(file_path)
       # compute_disparity_and_distance(lfile_path, rfile_path)
        detect_fruit_and_draw_bounding_boxes(file_path)

# Create the main application window
window = tk.Tk()
window.title("Palm Oil Fruit Classifier")
window.geometry("800x600")
window.configure(bg="#2C2F33")

# Create a title label
title_label = Label(window, text="Palm Oil Fruit Classifier", font=("Arial", 24, "bold"), bg="#2C2F33", fg="#FFFFFF")
title_label.pack(pady=20)

# Create a frame for the upload button and image display
frame = Frame(window, bg="#2C2F33")
frame.pack(pady=10)

# Button to upload an image
upload_button = Button(frame, text="Upload Images", command=upload_image, font=("Arial", 16), bg="#7289DA", fg="white", padx=10, pady=5)
upload_button.pack(pady=10)

# Label to display the uploaded image
image_label = Label(frame, bg="#2C2F33", borderwidth=2, relief="groove")
image_label.pack(pady=10)

# Create a frame for the result labels
results_frame = Frame(window, bg="#2C2F33")
results_frame.pack(pady=20)

# Result labels with padding and alignment within the frame
result_label = Label(results_frame, text="Predicted Class: ", font=("Arial", 16), bg="#2C2F33", fg="#FFFFFF")
result_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

result_label1 = Label(results_frame, text="Predicted Weight: ", font=("Arial", 16), bg="#2C2F33", fg="#FFFFFF")
result_label1.grid(row=1, column=0, sticky="w", padx=10, pady=5)

#result_label2 = Label(results_frame, text="Predicted Distance: ", font=("Arial", 16), bg="#2C2F33", fg="#FFFFFF")
#result_label2.grid(row=2, column=0, sticky="w", padx=10, pady=5)

result_label4 = Label(results_frame, text="Predicted Price (MYR): ", font=("Arial", 16), bg="#2C2F33", fg="#FFFFFF")
result_label4.grid(row=3, column=0, sticky="w", padx=10, pady=5)

# Start the Tkinter event loop
window.mainloop()
