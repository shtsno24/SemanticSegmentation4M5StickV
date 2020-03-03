import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
device_list = device_lib.list_local_devices()

import Model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

try:
    # TRAIN_RECORDS = "./data/VOC2012_resize_train.npz"
    # TEST_RECORDS = "./data/VOC2012_resize_val.npz"
    TRAIN_RECORDS = "./VOC2012_resize_train.npz"
    TEST_RECORDS = "./VOC2012_resize_val.npz"
    BATCH_SIZE = 8
    SHUFFLE_SIZE = 100
    TRAIN_DATASET_SIZE = 1464 * 2
    TEST_DATASET_SIZE = 1450 * 2
    EPOCHS = 500
    LABELS = 21
    COLOR_DEPTH = 3
    CROP_HEIGHT = 128
    CROP_WIDTH = 160

    with tf.device('/cpu:0'):
        # Load data from .npz
        print("Load dataset...\n\n")
        with np.load(TRAIN_RECORDS) as f:
            image_data = f["image"]
            annotation_data = f["annotation"]
            label_balance_array_resize = f["label_pix_resize"]
            label_pixel_count_array_resize = f["label_pix_cnt_resize"]
        annotation_data = annotation_data.astype(np.float32)
        image_data = image_data.astype(np.float32)
        image_data /= 255.0

        image_freq = label_balance_array_resize / label_pixel_count_array_resize
        CLASS_WEIGHT = {i: (np.median(image_freq) / image_freq)[i] for i in range(LABELS)}
        SAMPLE_WEIGHT = np.array([[[CLASS_WEIGHT[w] for w in range(LABELS)] for rows in range(CROP_WIDTH)] for columns in range(CROP_HEIGHT)])
        SAMPLE_WEIGHT = SAMPLE_WEIGHT.reshape((1,) + SAMPLE_WEIGHT.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((image_data, annotation_data))
        train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).repeat(-1)

        print(CLASS_WEIGHT)
        print(train_dataset)

        with np.load(TEST_RECORDS) as f:
            image_data = f["image"]
            annotation_data = f["annotation"]
        annotation_data = annotation_data.astype(np.float32)
        image_data = image_data.astype(np.float32)
        image_data /= 255.0
        test_dataset = tf.data.Dataset.from_tensor_slices((image_data, annotation_data))
        test_dataset = test_dataset.batch(BATCH_SIZE).repeat(-1)
        print(test_dataset, "\n\nDone")

    # Load model
    print("Load Model...\n\n")
    model = Model.TestNet()
    model.summary()
    print("\nDone")

    try:
        # Train model
        print("\n\nTrain Model...")
        model.compile(loss=Model.weighted_SparseCategoricalCrossentropy(SAMPLE_WEIGHT), optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS,
                steps_per_epoch=int(TRAIN_DATASET_SIZE / BATCH_SIZE),
                validation_steps=int(TEST_DATASET_SIZE / BATCH_SIZE / 100))
        print("  Done\n\n")
    except:
        import traceback
        traceback.print_exc()

    try:
        # Save model
        print("\n\nSave Model...")
        model.save('TestNet_VOC2012_npz.h5')
        print("  Done\n\n")
    except:
        import traceback
        traceback.print_exc()

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")
