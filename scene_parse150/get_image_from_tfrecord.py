import os
import sys
import csv
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image

def get_class_weight(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        l.pop(0)
        l_T = [list(x) for x in zip(*l)]
        out_list = [1.0 - float(x) for x in l_T[1]]
        out_list.insert(0, 0.0)
    return out_list

TFRECORD_NAME = "./data/scene_parse150_resize_test.tfrecords"
CLASS_WEIGHTING_FILE = "./data/objectinfo150.csv"
raw_image_dataset = tf.data.TFRecordDataset(TFRECORD_NAME)

height = 120
width = 160
color_depth = 3
show_index = np.random.randint(50)

features = {
    "annotation": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, features)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for i, image_features in enumerate(parsed_image_dataset):
    image_decoded = tf.io.decode_raw(image_features["image"], tf.uint8)
    image_data = tf.reshape(image_decoded, (height, width, color_depth))

    annotation_decoded = tf.io.decode_raw(image_features["annotation"], tf.uint8)
    annotation_data = tf.reshape(annotation_decoded, (height, width, 1))

    if i == show_index:
        image = image_data.numpy()
        annotation = annotation_data.numpy()
        annotation = np.concatenate((annotation, annotation, annotation), axis=2)
        output_data = np.concatenate((annotation, image), axis=1)
        image_object = Image.fromarray(output_data)
        image_object.show()
        print(get_class_weight(CLASS_WEIGHTING_FILE), len(get_class_weight(CLASS_WEIGHTING_FILE)))