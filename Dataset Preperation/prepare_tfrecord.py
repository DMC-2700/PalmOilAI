import tensorflow as tf
import os
import glob

def create_tfrecord(output_file, image_dir, label_map):
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(output_file) as writer:
        for label, class_name in label_map.items():
            # Create a path for the images of this class
            class_dir = os.path.join(image_dir, class_name)
            image_files = glob.glob(os.path.join(class_dir, '*.jpg'))  # Adjust if your images have different extensions

            for image_file in image_files:
                # Read the image file
                with tf.io.gfile.GFile(image_file, 'rb') as img_file:
                    img_data = img_file.read()

                # Create a TFRecord example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                    'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }))
                writer.write(example.SerializeToString())

# Example usage
if __name__ == "__main__":
    # Define paths
    image_dir = r'Dataset\Training\Palmoil'  # Path to the image directory
    output_file = r'Dataset\palmoil.tfrecord'  # Output TFRecord file

    # Create a label map for the classes
    label_map = {
        0: 'ripe',
        1: 'unripe',
        2: 'overripe'
    }

    create_tfrecord(output_file, image_dir, label_map)
    print("TFRecord file created successfully.")
