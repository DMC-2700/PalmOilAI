import carla
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image):
    """
    Preprocess the image to match the input requirements of the palm oil scanner model.
    Args:
        image (numpy.ndarray): The raw RGB image from the camera feed.

    Returns:
        numpy.ndarray: Preprocessed image ready for model input.
    """
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values to range [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

def main():
    """
    Main function to run the CARLA simulation with the palm oil scanner model.
    """
    # Load the pre-trained palm oil scanner model
    model = tf.keras.models.load_model('path_to_your_model')

    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Set a timeout for server connection

    # Get the simulation world and blueprint library
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Spawn a vehicle in the simulation
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # Choose the first vehicle blueprint
    spawn_point = world.get_map().get_spawn_points()[0]  # Choose the first spawn point
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)  # Spawn the vehicle

    # Attach a camera to the vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')  # Set camera resolution
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')  # Set field of view
    camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))  # Position camera on the vehicle
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def process_image(image):
        """
        Callback function to process images from the camera feed.
        Args:
            image (carla.Image): The raw image from CARLA's camera sensor.
        """
        # Convert CARLA raw image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # Reshape to (height, width, 4)
        rgb_image = array[:, :, :3]  # Drop the alpha channel

        # Preprocess and predict using the palm oil scanner model
        processed = preprocess_image(rgb_image)
        predictions = model.predict(processed)

        # Display predictions in the console
        print("Predictions:", predictions)

        # Show the live camera feed with OpenCV
        cv2.imshow("Camera Feed", rgb_image)
        cv2.waitKey(1)

    # Attach the callback to the camera sensor
    camera.listen(process_image)

    try:
        # Enable autopilot for the vehicle
        vehicle.set_autopilot(True)
        print("Simulation running. Press Ctrl+C to stop.")
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        print("Stopping simulation.")
    finally:
        # Clean up actors to avoid resource leaks
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
