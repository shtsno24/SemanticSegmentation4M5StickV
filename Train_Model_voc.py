import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import save_img

print(tf.version.VERSION)
BATCH_SIZE = 1
NUM_BOXES = 1
CHANNELS = 3
CROP_HEIGHT = 120
CROP_WIDTH = 160
SAMPLE = np.random.randint(500)

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
tf_dataset = tfds.load(name="voc/2012", batch_size=BATCH_SIZE)

dataset = tfds.as_numpy(tf_dataset)
dataset_train, dataset_test = dataset["train"], dataset["test"]

# Build your input pipeline

for i in range(SAMPLE):
    features = next(dataset_test)
    image, label = features["image"], features["labels"]
    print("image : ", image.shape, image.dtype)
    print("label : ", label.shape, label.dtype)
    if i == SAMPLE / 2:
        save_img("image_raw.png", image[0])
        save_img("label_raw.png", label[0])

    boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, image.shape[2], image.shape[2])
    image = tf.image.resize_with_pad(image, CROP_HEIGHT, CROP_WIDTH)
    image = tf.cast(image, tf.uint8)
    label = tf.image.resize_with_crop_or_pad(label, image.shape[2], image.shape[2])
    label = tf.image.resize_with_pad(label, CROP_HEIGHT, CROP_WIDTH)
    label = tf.cast(label, tf.uint8)

    print("image : ", image.shape, image.dtype)
    print("annotation : ", label.shape, label.dtype)

    print("\n\n")
    if i == SAMPLE / 2:
        save_img("image_cropped.png", image[0])
        save_img("annotation_cropped.png", label[0])
