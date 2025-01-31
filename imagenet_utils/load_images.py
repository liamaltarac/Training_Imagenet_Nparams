
import tensorflow as tf

def load_images(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label