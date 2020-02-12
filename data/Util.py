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


def label2onehot(a, axis=0, DTYPE=np.uint8, labels=0):
    # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495 @ Approach #2
    if labels is 0:
        ncols = a.max()+1
    else:
        ncols = labels
    grid = np.ogrid[tuple(map(slice, a.shape))]
    grid.insert(axis, a)

    out = np.zeros(a.shape + (ncols,), dtype=DTYPE)
    out[tuple(grid)] = 1
    return out


if __name__ == "__main__":
    index_palette = create_scene_parse150_label_dict(FILE_PATH, OBJECT_FILE)
    print(index_palette)
    x = np.random.randint(0, 150, (112, 112), dtype=np.uint8)
    y = label2onehot(x, axis=2)
    print("array_match : ", np.array_equal(x, y.argmax(axis=2)))
    print("shape x array : ", x.shape, " shape y array : ", y.shape)
