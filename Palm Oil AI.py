# Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Define Directories
train_dir = r"Dataset\Training\Palmoil"  # Update this path
val_dir = r"Dataset\Verification"  # Update this path

# Create ImageDataGenerator Instances
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
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

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

# Build and Compile the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: ripe, unripe, overripe
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
        epochs=28,  # Adjust as needed
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
else:
    history = model.fit(
        train_generator,
        epochs=28,  # Adjust as needed
        steps_per_epoch=train_generator.samples // train_generator.batch_size
    )

# Save the trained model
model.save('best_model.h5')  # Save the model for future use

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

# Specify the path to your test image
test_image_path = r"Dataset/example image.jpg"  # Update this path

# Load and preprocess the image
test_image = load_and_preprocess_image(test_image_path)

# Make predictions
predictions = model.predict(test_image)

# Get the predicted class
predicted_class = np.argmax(predictions[0])

# Map predicted class index to class names
class_names = ['ripe', 'unripe', 'overripe']  # Adjust based on your dataset
predicted_label = class_names[predicted_class]

# Output the result
print(f"The predicted class for the image is: {predicted_label}")
