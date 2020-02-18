import numpy as np
import tensorflow as tf
from tensorflow import keras

import Model


def parse_tfrecords(example, DTYPE=tf.float32):

    features = tf.io.parse_single_example(example,
                                          features={
                                          'height': tf.io.FixedLenFeature([], tf.int64),
                                          'width': tf.io.FixedLenFeature([], tf.int64),
                                          'annotation': tf.io.FixedLenFeature([], tf.string),
                                          'image': tf.io.FixedLenFeature([], tf.string),
                                          })

    height = 112
    width = 112
    labels = 151
    color_depth = 3

    annotation_decoded = tf.io.decode_raw(features["annotation"], tf.uint8)
    image_decoded = tf.io.decode_raw(features["image"], tf.uint8)

    return image_decoded, annotation_decoded


try:
    # PATH = "data/scene_parse150_resize.npz"
    # PATH = "data/scene_parse150_resize_test.npz"

    TRAIN_RECORDS = "./data/scene_parse150_resize_train.tfrecords"
    TEST_RECORDS = "./data/scene_parse150_resize_test.tfrecords"
    BATCH_SIZE = 128
    SHUFFLE_SIZE = 100
    EPOCHS = 10
    LABELS = 151
    COLOR_DEPTH = 3
    CROP_HEIGHT = 112
    CROP_WIDTH = 112

    # # Load data from .npz
    # print("\n\nLoad data...", end="")
    # with np.load(PATH) as data:
    #     train_image = data["train_image"]
    #     train_annotation = data["train_annotation"]
    #     test_image = data["test_image"]
    #     test_annotation = data["test_annotation"]
    # print("  Done\n\n")

    # Load data from .tfrecords
    train_dataset = tf.data.TFRecordDataset(TEST_RECORDS)

    train_dataset = train_dataset.map(parse_tfrecords)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat(-1)
    train_iterator = train_dataset.make_one_shot_iterator()

    train_images, train_annotations = train_iterator.get_next()
    train_annotations = tf.cast(train_annotations, dtype=tf.float32) / 255.0
    train_images = tf.cast(train_images, dtype=tf.float32) / 255.0
    train_annotations = tf.reshape(train_annotations, (-1, CROP_HEIGHT, CROP_WIDTH, LABELS))
    train_images = tf.reshape(train_images, (-1, CROP_HEIGHT, CROP_WIDTH, COLOR_DEPTH))


    test_dataset = tf.data.TFRecordDataset(TEST_RECORDS)

    test_dataset = test_dataset.map(parse_tfrecords)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.repeat(-1)
    test_iterator = test_dataset.make_one_shot_iterator()

    test_images, test_annotations = test_iterator.get_next()
    test_annotations = tf.cast(test_annotations, dtype=tf.float32) / 255.0
    test_images = tf.cast(test_images, dtype=tf.float32) / 255.0
    test_annotations = tf.reshape(test_annotations, (-1, CROP_HEIGHT, CROP_WIDTH, LABELS))
    test_images = tf.reshape(test_images, (-1, CROP_HEIGHT, CROP_WIDTH, COLOR_DEPTH))

    print(train_images)

    # image_feature_description = {
    #     'height': tf.io.FixedLenFeature([], tf.int64),
    #     'width': tf.io.FixedLenFeature([], tf.int64),
    #     'annotation': tf.io.FixedLenFeature([], tf.string),
    #     'image': tf.io.FixedLenFeature([], tf.string),
    # }

    # Generate dataset
    # print("Generate dataset...", end="")
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_annotation))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_annotation))

    # train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    # print("  Done\n\n")

    # Define model
    print("Load Model...\n\n")
    model = Model.TestNet()
    model.summary()
    print("\n\nDone\n\n")

    # Train model
    print("Train Model...", end="")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)
    model.save('TestNet.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")