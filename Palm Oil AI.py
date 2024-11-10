# Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image  # Import OpenCV for stereo depth estimation

# Define directories
train_dir = r"Dataset/Training"  # This should now point to the parent directory containing both Palmoil and Non_Palmoil
val_dir = r"Dataset/Verification"  # Update this path if you have a validation set

def is_valid_image(file_path):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return file_path.lower().endswith(valid_extensions)

is_valid_image(train_dir)
is_valid_image(val_dir)
    
def verify_images_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if it’s an actual image
            except (IOError, SyntaxError):
                print(f"Corrupted or unreadable image file: {file_path}")

def detect_fruit_and_draw_bounding_boxes(image_path, size_to_weight_ratio=1):
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
    
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the size (area) and estimate the weight
        area = w * h  # Approximate area of the bounding box
        estimated_weight = area * size_to_weight_ratio  # Adjust ratio as needed

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Weight: {estimated_weight:.2f} kg", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Palm Fruit Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Creating Data Generator Instances
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Pointing to the 'Dataset' directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Classes detected: {train_generator.class_indices}")

# Check validation directory
if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
    print("Validation directory is empty or does not exist. Skipping validation.")
    validation_generator = None
else:
    print("Validation directory found. Classes:")
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name}: {len(os.listdir(class_path))} images")
    # Create validation generator
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

# your mom
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes: Unripe, Overripe, Underipe, Non_Palmoil
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
if validation_generator:  # Only train if validation generator exists
    history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
else:
    history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=train_generator.samples // train_generator.batch_size
    )

# Save the trained model
model.save('best_model.h5')

# Plot Training History
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
if validation_generator:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
if validation_generator:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

# Test a Single Image
def load_and_preprocess_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Classify and Predict Image Label
def classify_image(img_path):
    # Load and preprocess the image
    test_image = load_and_preprocess_image(img_path)
    # Make predictions
    predictions = model.predict(test_image)
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    # Map predicted class index to class names
    class_names = ['Unripe', 'Overripe', 'Ripe', 'Non_Palmoil']
    predicted_label = class_names[predicted_class]
    print(f"The predicted class for the image is: {predicted_label}")

# Example usage for classification
test_image_path = r"Dataset/example image.jpg"  # Update this path
classify_image(test_image_path)

# Stereo Depth Estimation Parameters
focal_length = 800  # Replace with your camera's focal length in pixels
baseline = 0.06     # Distance between cameras in meters (replace with actual)

def compute_distance(focal_length, baseline, disparity_value):
    if disparity_value > 0:
        return (focal_length * baseline) / disparity_value
    else:
        return None  # Return None for invalid disparity values

def compute_disparity_and_distance(left_image_path, right_image_path):
    # Load stereo images
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded
    if left_image is None or right_image is None:
        raise Exception("Could not load one or both images. Check the paths.")

    # Preprocessing: Apply Gaussian Blur to reduce noise
    left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
    right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

    # StereoSGBM parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 9,  # Must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=32
    )

    # Compute disparity
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

    # Post-processing: Apply a median filter to smooth the disparity map
    disparity_map = cv2.medianBlur(disparity_map, 5)

    # Display the disparity map
    cv2.imshow("Disparity Map", disparity_map / disparity_map.max())  # Normalize for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate distance based on disparity at the center point
    center = (disparity_map.shape[1] // 2, disparity_map.shape[0] // 2)
    disparity_value = disparity_map[center[1], center[0]]

    # Use compute_distance function for distance calculation
    distance = compute_distance(focal_length, baseline, disparity_value)
    
    if distance:
        print(f"Estimated distance to object at center: {distance:.2f} meters")
    else:
        print("Invalid disparity; unable to estimate distance.")

# Having A Stroke

#Define Required Paths For Distance Estimation
left_image_path = r"Dataset/left.jpeg"
right_image_path = r"Dataset/right.jpeg"

#Run Functions for distance/boundingboxes
compute_disparity_and_distance(left_image_path, right_image_path)
detect_fruit_and_draw_bounding_boxes(r"Dataset/example image.jpg")