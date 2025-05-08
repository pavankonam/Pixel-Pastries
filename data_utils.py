import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from config import DATA_DIR, BATCH_SIZE

def load_images():
    #performing the Preprocessing of images
    image_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    #resizing images to 64*64 and ensuring 3 channels (RGB)
    images = [tf.image.resize(tf.image.decode_image(tf.io.read_file(p), channels=3), (64, 64)) for p in image_paths]
    images = tf.stack(images)
    images = (tf.cast(images, tf.float32) - 127.5) / 127.5
    return images.numpy()

def prepare_datasets(all_images):
    #Splitting the data 
    train_images, test_images = train_test_split(all_images, test_size=0.1, random_state=42)#splitting 10% data as test
    train_images, val_images = train_test_split(train_images, test_size=0.1, random_state=42)#splitting remaining into train and validation
    print(f"Train images: {train_images.shape}")
    print(f"Validation images: {val_images.shape}")
    print(f"Test images: {test_images.shape}")
    return (
        tf.data.Dataset.from_tensor_slices(train_images).shuffle(1000).batch(BATCH_SIZE),
        tf.convert_to_tensor(val_images),
        tf.convert_to_tensor(test_images)
    )