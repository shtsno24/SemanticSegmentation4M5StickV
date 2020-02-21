import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras

import Model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

def parse_tfrecords(example, DTYPE=tf.float32):

    features = tf.io.parse_single_example(example,
                                          features={
                                          'image': tf.io.FixedLenFeature([], tf.string),
                                          'annotation': tf.io.FixedLenFeature([], tf.string)
                                          })

    height = 120
    width = 160
    color_depth = 3

    annotation_decoded = tf.io.decode_raw(features["annotation"], tf.uint8)
    image_decoded = tf.io.decode_raw(features["image"], tf.uint8)

    annotation_data = tf.reshape(annotation_decoded, (height, width))
    image_data = tf.reshape(image_decoded, (height, width, color_depth))

    annotation_data = tf.cast(annotation_data, dtype=DTYPE)
    image_data = tf.cast(image_data, dtype=DTYPE) / 255.0

    return image_data, annotation_data


try:
    # TRAIN_RECORDS = "./data/VOC2012_resize_trainval.tfrecords"
    # TEST_RECORDS = "./data/VOC2012_resize_val.tfrecords"
    TRAIN_RECORDS = "./VOC2012_resize_trainval.tfrecords"
    TEST_RECORDS = "./VOC2012_resize_val.tfrecords"
    BATCH_SIZE_TRAIN = 15
    BATCH_SIZE_TEST = 10
    SHUFFLE_SIZE = 100
    TRAIN_DATASET_SIZE = 2913
    TEST_DATASET_SIZE = 1449
    EPOCHS = 1000
    LABELS = 21
    COLOR_DEPTH = 3
    CROP_HEIGHT = 120
    CROP_WIDTH = 160

    # Load data from .tfrecords
    print("Load dataset...\n\n")
    train_dataset = tf.data.TFRecordDataset(TRAIN_RECORDS)
    train_dataset = train_dataset.map(parse_tfrecords)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE_TRAIN)
    train_dataset = train_dataset.repeat(-1)

    test_dataset = tf.data.TFRecordDataset(TEST_RECORDS)
    test_dataset = test_dataset.map(parse_tfrecords)
    test_dataset = test_dataset.batch(BATCH_SIZE_TEST)
    test_dataset = test_dataset.repeat(-1)

    print(train_dataset, "\n\nDone")

    # Load model
    print("Load Model...\n\n")
    model = Model.TestNet()
    model.summary()
    print("\nDone")

    # Train model
    print("\n\nTrain Model...")
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adadelta', metrics=["accuracy"])
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS,
              steps_per_epoch=int(TRAIN_DATASET_SIZE/BATCH_SIZE_TRAIN),
              validation_steps=int(TEST_DATASET_SIZE/BATCH_SIZE_TEST))
    model.save('TestNet_VOC2012.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")