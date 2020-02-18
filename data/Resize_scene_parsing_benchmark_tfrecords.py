import os
import sys
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image
from Util import label2onehot, create_scene_parse150_label_dict, get_file_list_from_directory

try:
    tf.add(1, 1) # To show INFO
    CROP_HEIGHT = 112
    CROP_WIDTH = 112
    LABELS = 151
    COLOR_DEPTH = 3
    SAMPLE = np.random.randint(30)

    FILE_PATH = "./color150/"
    OBJECT_FILE = "objectinfo150.csv"
    INDEX_PALETTE = create_scene_parse150_label_dict(FILE_PATH, OBJECT_FILE)

    TRAIN_RECORDS = "scene_parse150_resize_train.tfrecords"
    TEST_RECORDS = "scene_parse150_resize_test.tfrecords"

    dataset_lists = {"train_image": [], "train_annotation": [], "test_image": [], "test_annotation": []}
    image_file_directories = {"train_image": ("./ADEChallengeData2016/images/training/", []),
                              "train_annotation": ("./ADEChallengeData2016/annotations/training/", []),
                              "test_image": ("./ADEChallengeData2016/images/validation/", []),
                              "test_annotation": ("./ADEChallengeData2016/annotations/validation/", []),
                              "data_num": {"train_image": 0, "train_annotation": 0, "test_image": 0, "test_annotation": 0}}

    # Get File Name
    try:
        print("\n\n\nGet File Name...")
        for Mode in ("train_image", "train_annotation", "test_image", "test_annotation"):
        # for Mode in ("test_image", "test_annotation"):
            file_list, data_num = get_file_list_from_directory(image_file_directories[Mode][0])
            image_file_directories[Mode][1].append(file_list)
            image_file_directories["data_num"][Mode] = data_num
            print(Mode, data_num)
        print("\nDone")
    except:
        traceback.print_exc()
        sys.exit(1)


    # Generate Test Data Records
    try:
        print("\n\n\nGenerate Test Data Records...\n")
        with tf.io.TFRecordWriter(TEST_RECORDS) as writer:
            data_num = image_file_directories["data_num"]["test_image"]
            show_flag = 0
            for i in range(data_num):
                # Loading image-like data
                annotation_file_name = image_file_directories["test_annotation"][0] + image_file_directories["test_annotation"][1][0][i]
                image_file_name = image_file_directories["test_image"][0] + image_file_directories["test_image"][1][0][i]
                # print("loading :",annotation_file_name)
                # print("loading :",image_file_name)
                with Image.open(annotation_file_name) as annotation_object:
                    annotation_data = np.array(annotation_object, dtype=np.uint8)
                    annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                with Image.open(image_file_name) as image_object:
                    image_object = image_object.convert("RGB")
                    image_data = np.array(image_object, dtype=np.uint8)

                # Resizeing data
                X, Y = image_data.shape[0], image_data.shape[1]
                if X < Y:
                    long_side = Y
                else:
                    long_side = X
                annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                annotation_data = tf.image.resize(annotation_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                image_data = tf.image.resize(image_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # Convert annotation_data to one-hot array
                annotation_data = annotation_data.numpy()
                annotation_data = annotation_data.reshape(annotation_data.shape[:2])
                annotation_data = tf.one_hot(annotation_data, depth=LABELS)

                # Convert ndarray to raw format                
                annotation_data = annotation_data.numpy().tostring()
                image_data = image_data.numpy().tostring()

                # Write to Record
                feature = {
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_HEIGHT])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_WIDTH])),
                    "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[LABELS])),
                    "color_depth" : tf.train.Feature(int64_list=tf.train.Int64List(value=[COLOR_DEPTH])),
                    "annotation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_data])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                status = int((i + 1) / data_num * 100)
                if status % 2 == 0:
                    if show_flag == 0:
                        show_flag = 1
                        print("\033[F", "■" * int(status / 2), int(status), "%")  
                else:
                    show_flag = 0

        print("\n\nDone")
    except:
        traceback.print_exc()
        sys.exit(2)


    # Generate Train Data Records
    try:
        print("\n\n\nGenerate Train Data Records...\n")
        with tf.io.TFRecordWriter(TRAIN_RECORDS) as writer:
            data_num = image_file_directories["data_num"]["train_image"]
            show_flag = 0
            for i in range(data_num):
                # Loading image-like data
                annotation_file_name = image_file_directories["train_annotation"][0] + image_file_directories["train_annotation"][1][0][i]
                image_file_name = image_file_directories["train_image"][0] + image_file_directories["train_image"][1][0][i]
                # print("loading :",annotation_file_name)
                # print("loading :",image_file_name)
                with Image.open(annotation_file_name) as annotation_object:
                    annotation_data = np.array(annotation_object, dtype=np.uint8)
                    annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                with Image.open(image_file_name) as image_object:
                    image_object = image_object.convert("RGB")
                    image_data = np.array(image_object, dtype=np.uint8)

                # Resizeing data
                X, Y = image_data.shape[0], image_data.shape[1]
                if X < Y:
                    long_side = Y
                else:
                    long_side = X
                annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                annotation_data = tf.image.resize(annotation_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                image_data = tf.image.resize(image_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # Convert annotation_data to one-hot array
                annotation_data = annotation_data.numpy()
                annotation_data = annotation_data.reshape(annotation_data.shape[:2])
                annotation_data = tf.one_hot(annotation_data, depth=LABELS)

                # Convert ndarray to raw format                
                annotation_data = annotation_data.numpy().tostring()
                image_data = image_data.numpy().tostring()

                # Write to Record
                feature = {
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_HEIGHT])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_WIDTH])),
                    "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[LABELS])),
                    "color_depth" : tf.train.Feature(int64_list=tf.train.Int64List(value=[COLOR_DEPTH])),
                    "annotation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_data])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                status = int((i + 1) / data_num * 100)
                if status % 2 == 0:
                    if show_flag == 0:
                        show_flag = 1
                        print("\033[F", "■" * int(status / 2), int(status), "%")  
                else:
                    show_flag = 0

        print("\n\nDone")
    except:
        traceback.print_exc()
        sys.exit(3)

    print("\n\n\nAll Resizing Done!!!")

except:
    import traceback
    traceback.print_exc()


finally:
    input(">>>")
