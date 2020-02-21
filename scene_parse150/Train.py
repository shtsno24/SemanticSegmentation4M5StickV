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


def get_class_weight(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        l.pop(0)
        l_T = [list(x) for x in zip(*l)]
        out_list = [1.0 - float(x) for x in l_T[1]]
        out_list.insert(0, 0.0)
    return out_list

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
    # TRAIN_RECORDS = "./data/scene_parse150_resize_train.tfrecords"
    # TEST_RECORDS = "./data/scene_parse150_resize_test.tfrecords"
    # CLASS_WEIGHTING_FILE = "./data/objectinfo150.csv"
    CLASS_WEIGHTING_FILE = "./objectinfo150.csv"
    TRAIN_RECORDS = "./scene_parse150_resize_train.tfrecords"
    TEST_RECORDS = "./scene_parse150_resize_test.tfrecords"
    BATCH_SIZE_TRAIN = 10 # 47 or 43
    BATCH_SIZE_TEST = 100
    SHUFFLE_SIZE = 100
    TRAIN_DATASET_SIZE = 20210
    TEST_DATASET_SIZE = 2000
    EPOCHS = 10
    LABELS = 151
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
    class_weighting = get_class_weight(CLASS_WEIGHTING_FILE)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=["accuracy"])
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS,
              steps_per_epoch=int(TRAIN_DATASET_SIZE/BATCH_SIZE_TRAIN),
              validation_steps=int(TEST_DATASET_SIZE/BATCH_SIZE_TEST),
              class_weight=class_weighting)
    model.save('TestNet_scene150.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")