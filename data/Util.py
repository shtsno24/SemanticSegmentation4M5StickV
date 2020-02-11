import numpy as np
from PIL import Image
import os
import csv

FILE_PATH = "./color150/"
OBJECT_FILE = "objectinfo150.csv"


def get_index_name(file_name):
    index_dict = {0: "ignore"}
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            index_dict.update({int(row[0]): row[5].split(";")[0]})
    return index_dict


def generate_color_index(path_to_file, index_dict, file_format="jpg"):

    index_palette = {}
    for i in index_dict.keys():
        img = Image.open(path_to_file + index_dict[i] + "." + file_format)
        img_array = np.asarray(img, dtype=np.uint8)
        rgb_tuple = (img_array[0][0][0], img_array[0][0][1], img_array[0][0][2])

        index_palette.update({i: (rgb_tuple, index_dict[i])})

    return index_palette


def create_scene_parse150_label_dict(path_to_file, object_file, file_format="jpg"):
    """
    Creates a label colormap and label names used in Scene Parse150 segmentation benchmark.
    https://github.com/CSAILVision/sceneparsing/tree/master/visualizationCode/color150
    """
    index_dict = get_index_name(object_file)
    index_palette = generate_color_index(path_to_file, index_dict, file_format=file_format)

    return index_palette


def label2onehot(x, index_palette, DTYPE=np.uint8, print_log=False):
    classes = len(index_palette)
    shapes = x.shape[:2] + (classes,)

    output_array = np.zeros(shapes, dtype=DTYPE)
    for X in range(output_array.shape[0]):
        for Y in range(output_array.shape[1]):
            output_array[X][Y][x[X][Y][0]] = 1
            if print_log is True and x[X][Y][0] != 0:
                print(x[X][Y], int(float(x[X][Y][0] / 10.0) * 256 + x[X][Y][1]), index_palette[x[X][Y][0]])
    return output_array.astype(DTYPE)


if __name__ == "__main__":
    index_palette = create_scene_parse150_label_dict(FILE_PATH, OBJECT_FILE)
    print(index_palette)
    x = np.zeros((112, 112, 3), dtype=np.uint8)
    y = label2onehot(x, index_palette)
    print(y.shape)
