import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import save_img

try:
    print(tf.version.VERSION)
    BATCH_SIZE = 1
    NUM_BOXES = 1
    CHANNELS = 3
    CROP_HEIGHT = 112
    CROP_WIDTH = 112
    SAMPLE = np.random.randint(2000)

    # See available datasets
    print(tfds.list_builders())

    # Construct a tf.data.Dataset
    tf_dataset = tfds.load(name="scene_parse150")

    dataset = tfds.as_numpy(tf_dataset)
    # dataset_train, dataset_test = dataset["train"], dataset["test"]
    # dataset_image_list, dataset_annotation_list = [], []
    dataset_lists = {"train_image": [], "train_annotation": [], "test_image": [], "test_annotation": []}

    # Build your input pipeline

    print("\n\n\nResizing Images...")
    for Mode in ("train", "test"):
        i = 0
        for features in dataset[Mode]:
            image, annotation = features["image"], features["annotation"]
            height, width = image.shape[0], image.shape[1]
            # print("image : ", image.shape, image.dtype)
            # print("annotation : ", annotation.shape, annotation.dtype)
            if i == SAMPLE:
                # print("================================")
                save_img("image_raw.png", image)
                save_img("annotation_raw.png", annotation)

            if height > width:
                resize_length = height
            else:
                resize_length = width

            image = tf.image.resize_with_crop_or_pad(image, resize_length, resize_length)
            image = tf.image.resize_with_pad(image, CROP_HEIGHT, CROP_WIDTH)
            image = tf.cast(image, tf.uint8)
            annotation = tf.image.resize_with_crop_or_pad(annotation, resize_length, resize_length)
            annotation = tf.image.resize_with_pad(annotation, CROP_HEIGHT, CROP_WIDTH)
            annotation = tf.cast(annotation, tf.uint8)

            # print("image : ", image.dtype, image.shape, image.dtype)
            # print("annotation : ", annotation.dtype, annotation.shape, annotation.dtype)

            if i == SAMPLE:
                save_img("image_resized.png", image)
                save_img("annotation_resized.png", annotation)

            dataset_lists[Mode + "_image"].append(image)
            dataset_lists[Mode + "_annotation"].append(annotation)
            i += 1

    print("\n\n\n")
    print("train_image :", len(dataset_lists["train_image"]), dataset_lists["train_image"][0].shape)
    print("train_annotation :", len(dataset_lists["train_annotation"]), dataset_lists["train_annotation"][0].shape)
    print("test_image :", len(dataset_lists["test_image"]), dataset_lists["test_image"][0].shape)
    print("test_annotation :", len(dataset_lists["test_annotation"]), dataset_lists["test_annotation"][0].shape)

    print("\n\n\nConvert to ndarray...   ", end="")

    train_image_array = np.asarray(dataset_lists["train_image"], dtype=np.uint8)
    train_annotation_array = np.asarray(dataset_lists["train_annotation"], dtype=np.uint8)
    test_image_array = np.asarray(dataset_lists["test_image"], dtype=np.uint8)
    test_annotation_array = np.asarray(dataset_lists["test_annotation"], dtype=np.uint8)

    print("Done")

    np.savez("data/scene_parse150_resize", 
            train_image=train_image_array, 
            train_annotation=train_annotation_array, 
            test_image=test_image_array, 
            test_annotation=test_annotation_array)

    print("Saved")

except:
    import traceback
    traceback.print_exc()

finally:
    print(">>")
