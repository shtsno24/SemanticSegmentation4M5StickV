import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import save_img, array_to_img
from Util import label2onehot, create_scene_parse150_label_dict

try:
    print(tf.version.VERSION)
    BATCH_SIZE = 1
    NUM_BOXES = 1
    CHANNELS = 3
    CROP_HEIGHT = 112
    CROP_WIDTH = 112
    # SAMPLE = np.random.randint(30)
    SAMPLE = 0
    FILE_PATH = "./color150/"
    OBJECT_FILE = "objectinfo150.csv"
    INDEX_PALETTE = create_scene_parse150_label_dict(FILE_PATH, OBJECT_FILE)

    # See available datasets
    print(tfds.list_builders())

    # Construct a tf.data.Dataset
    tf_dataset = tfds.load(name="scene_parse150")

    dataset = tfds.as_numpy(tf_dataset)
    print(tf_dataset)
    dataset_lists = {"train_image": [], "train_annotation": [], "test_image": [], "test_annotation": []}

    # # Build your input pipeline
    print("\n\n\nResizing Images...")
    # for Mode in ("train", "test"):
    for Mode in ("test",):
        i = 0
        for features in dataset[Mode]:
            image, annotation = features["image"], features["annotation"]
            height, width = image.shape[0], image.shape[1]
            if i == SAMPLE:
                print(image.shape, annotation.shape)
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
            annotation = tf.image.resize_with_pad(annotation, CROP_HEIGHT, CROP_WIDTH, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            annotation = tf.cast(annotation, tf.uint8)

            if i == SAMPLE:
                save_img("image_resized.png", image)
                save_img("annotation_resized.png", annotation)
                Shape = annotation.shape[:2] + (3,)

                one_hot_img_R = label2onehot(annotation.numpy(), INDEX_PALETTE, print_log=True)
                print(one_hot_img_R.shape)
                one_hot_img = np.zeros(Shape, dtype=np.uint8)
                one_hot_img_R = np.argmax(one_hot_img_R, axis=2)
                for X in range(Shape[0]):
                    for Y in range(Shape[1]):
                        one_hot_img[X][Y][0] = INDEX_PALETTE[one_hot_img_R[X][Y]][0][0]
                        one_hot_img[X][Y][1] = INDEX_PALETTE[one_hot_img_R[X][Y]][0][1]
                        one_hot_img[X][Y][2] = INDEX_PALETTE[one_hot_img_R[X][Y]][0][2]

                save_img("annotation_resized_one_hot.png", one_hot_img)

            annotation = label2onehot(annotation.numpy(), INDEX_PALETTE)
            dataset_lists[Mode + "_image"].append(image.numpy())
            dataset_lists[Mode + "_annotation"].append(annotation)
            i += 1

    print("\n\n\n")
    # print("train_image :", len(dataset_lists["train_image"]), dataset_lists["train_image"][0].shape, type(dataset_lists["train_image"][0]))
    # print("train_annotation :", len(dataset_lists["train_annotation"]), dataset_lists["train_annotation"][0].shape, type(dataset_lists["train_annotation"][0]))
    print("test_image :", len(dataset_lists["test_image"]), dataset_lists["test_image"][0].shape, type(dataset_lists["test_image"][0]))
    print("test_annotation :", len(dataset_lists["test_annotation"]), dataset_lists["test_annotation"][0].shape, type(dataset_lists["test_annotation"][0]))

    print("\n\n\nConvert to ndarray...   ", end="")

    # train_image_array = np.asarray(dataset_lists["train_image"], dtype=np.uint8)
    # train_annotation_array = np.asarray(dataset_lists["train_annotation"], dtype=np.uint8)
    test_image_array = np.asarray(dataset_lists["test_image"], dtype=np.uint8)
    test_annotation_array = np.asarray(dataset_lists["test_annotation"], dtype=np.uint8)

    print("Done")

    np.savez("scene_parse150_resize",
             # train_image=train_image_array,
             # train_annotation=train_annotation_array,
             test_image=test_image_array,
             test_annotation=test_annotation_array)

    print("Saved")

except:
    import traceback
    traceback.print_exc()

finally:
    print(">>")
