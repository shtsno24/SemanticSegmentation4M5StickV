import numpy as np
import tensorflow as tf
from tensorflow import keras

import Model

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)
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
                                          'annotation': tf.io.FixedLenFeature([], tf.string),
                                          'image': tf.io.FixedLenFeature([], tf.string),
                                          })

    height = 112
    width = 112
    color_depth = 3

    annotation_decoded = tf.io.decode_raw(features["annotation"], tf.uint8)
    image_decoded = tf.io.decode_raw(features["image"], tf.uint8)

    annotation_data = tf.reshape(annotation_decoded, (height, width))
    image_data = tf.reshape(image_decoded, (height, width, color_depth))

    annotation_data = tf.cast(annotation_data, dtype=DTYPE)
    image_data = tf.cast(image_data, dtype=DTYPE) / 255.0

    return image_data, annotation_data


try:
    # PATH = "data/scene_parse150_resize.npz"
    # PATH = "data/scene_parse150_resize_test.npz"

    # TRAIN_RECORDS = "./data/scene_parse150_resize_train.tfrecords"
    # TEST_RECORDS = "./data/scene_parse150_resize_test.tfrecords"
    TRAIN_RECORDS = "./scene_parse150_resize_train.tfrecords"
    TEST_RECORDS = "./scene_parse150_resize_test.tfrecords"
    BATCH_SIZE_TRAIN = 47 * 2 # or 43
    # BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 100
    SHUFFLE_SIZE = 100
    TRAIN_DATASET_SIZE = 20210
    TEST_DATASET_SIZE = 2000
    EPOCHS = 1000
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
    train_dataset = tf.data.TFRecordDataset(TRAIN_RECORDS)
    train_dataset = train_dataset.map(parse_tfrecords)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE_TRAIN)
    train_dataset = train_dataset.repeat(10)

    test_dataset = tf.data.TFRecordDataset(TEST_RECORDS)
    test_dataset = test_dataset.map(parse_tfrecords)
    test_dataset = test_dataset.batch(BATCH_SIZE_TEST)
    test_dataset = test_dataset.repeat(10)

    print(train_dataset)

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
    print("\nDone")

    # Train model
    print("\n\nTrain Model...")
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adadelta', metrics=["accuracy"])
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, steps_per_epoch=int(TRAIN_DATASET_SIZE/BATCH_SIZE_TRAIN), validation_steps=int(TEST_DATASET_SIZE/BATCH_SIZE_TEST))
    model.save('TestNet.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")