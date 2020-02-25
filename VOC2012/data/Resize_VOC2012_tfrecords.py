import os
import csv
import sys
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image
from Util import get_file_list_from_directory

try:
    tf.add(1, 1) # To show INFO
    CROP_HEIGHT = 120
    CROP_WIDTH = 160
    LABELS = 21
    COLOR_DEPTH = 3
    SAMPLE = np.random.randint(30)

    TRAIN_RECORDS = "VOC2012_resize_train.tfrecords"
    TRAINVAL_RECORDS = "VOC2012_resize_trainval.tfrecords"
    VAL_RECORDS = "VOC2012_resize_val.tfrecords"

    TRAIN_DATA_NAME = "./ImageSets/Segmentation/train.txt"
    TRAINVAL_DATA_NAME = "./ImageSets/Segmentation/trainval.txt"
    VAL_DATA_NAME = "./ImageSets/Segmentation/val.txt"

    image_file_directories = {"image": "./JPEGImages/", "annotation": "./SegmentationClass/"}

    # Generate Train Data Records
    try:
        print("\n\n\nGenerate Train Data Records...\n")
        with tf.io.TFRecordWriter(TRAIN_RECORDS) as writer:
            with open(TRAIN_DATA_NAME) as f:
                reader = csv.reader(f)
                show_flag = 0
                data_num = 1463
                for i, name in enumerate(reader):
                    # Loading image-like data
                    image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                    annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                    with Image.open(image_file_name) as image_object:
                        image_object.convert("RGB")
                        image_data = np.array(image_object, dtype=np.uint8)
                    with Image.open(annotation_file_name) as annotation_object:
                        annotation_data = np.array(annotation_object, dtype=np.uint8)
                        annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                        annotation_data[annotation_data == 255] = 0
                        if i == 0:
                            palette = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)
                            palette[21] = 255
                    # Resizeing data
                    X, Y = image_data.shape[0], image_data.shape[1]
                    if X < Y:
                        long_side = Y
                    else:
                        long_side = X
                    annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                    annotation_data = tf.image.resize(annotation_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                    image_data = tf.image.resize(image_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.LANCZOS5)

                    # Reshape annotation_data
                    annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])
                    annotation_data_buffer = annotation_data.numpy()

                    # Convert ndarray to raw format                
                    annotation_data = annotation_data.numpy().tobytes()
                    image_data = image_data.numpy().tobytes()

                    # Write to Record
                    feature = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        "annotation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_data]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())


                    status = int((i + 1) / data_num * 100)
                    if status % 2 == 0:
                        if show_flag == 0:
                            show_flag = 1
                            print("\033[F", int(status), "%|", "■" * int(status / 2))  
                    else:
                        show_flag = 0

                annotation_data = annotation_data_buffer
                annotation_image = Image.fromarray(annotation_data)
                annotation_image.putpalette(palette)
                annotation_image.show()

        print("\n\nDone")
    except:
        traceback.print_exc()
        sys.exit(1)

    # Generate Train Validate Data Records
    try:
        print("\n\n\nGenerate Train Validate Data Records...\n")
        with tf.io.TFRecordWriter(TRAINVAL_RECORDS) as writer:
            with open(TRAINVAL_DATA_NAME) as f:
                reader = csv.reader(f)
                show_flag = 0
                data_num = 2912
                for i, name in enumerate(reader):
                    # Loading image-like data
                    image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                    annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                    with Image.open(image_file_name) as image_object:
                        image_object.convert("RGB")
                        image_data = np.array(image_object, dtype=np.uint8)
                    with Image.open(annotation_file_name) as annotation_object:
                        annotation_data = np.array(annotation_object, dtype=np.uint8)
                        annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                        annotation_data[annotation_data == 255] = 0
                        if i == 0:
                            palette = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)
                            palette[21] = 255

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

                    # Reshape annotation_data
                    annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])

                    # Convert ndarray to raw format                
                    annotation_data = annotation_data.numpy().tobytes()
                    image_data = image_data.numpy().tobytes()

                    # Write to Record
                    feature = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        "annotation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_data]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                    status = int((i + 1) / data_num * 100)
                    if status % 2 == 0:
                        if show_flag == 0:
                            show_flag = 1
                            print("\033[F", int(status), "%|", "■" * int(status / 2))  
                    else:
                        show_flag = 0

        print("\n\nDone")
    except:
        traceback.print_exc()
        sys.exit(2)


    # Generate Validate Data Records
    try:
        print("\n\n\nGenerate Validate Data Records...\n")
        with tf.io.TFRecordWriter(VAL_RECORDS) as writer:
            with open(VAL_DATA_NAME) as f:
                reader = csv.reader(f)
                show_flag = 0
                data_num = 1448
                for i, name in enumerate(reader):
                    # Loading image-like data
                    image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                    annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                    with Image.open(image_file_name) as image_object:
                        image_object.convert("RGB")
                        image_data = np.array(image_object, dtype=np.uint8)
                    with Image.open(annotation_file_name) as annotation_object:
                        annotation_data = np.array(annotation_object, dtype=np.uint8)
                        annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                        annotation_data[annotation_data == 255] = 0
                        if i == 0:
                            palette = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)
                            palette[21] = 255

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

                    # Reshape annotation_data
                    annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])

                    # Convert ndarray to raw format                
                    annotation_data = annotation_data.numpy().tobytes()
                    image_data = image_data.numpy().tobytes()

                    # Write to Record
                    feature = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        "annotation": tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation_data]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                    status = int((i + 1) / data_num * 100)
                    if status % 2 == 0:
                        if show_flag == 0:
                            show_flag = 1
                            print("\033[F", int(status), "%|", "■" * int(status / 2))  
                    else:
                        show_flag = 0

        print("\n\nDone")
    except:
        traceback.print_exc()
        sys.exit(2)


    print("\n\n\nAll Resizing Done!!!")

except:
    import traceback
    traceback.print_exc()


finally:
    input(">>>")
