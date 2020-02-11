import numpy as np
from PIL import Image
import os

FILE_PATH = "./color150/"


def get_file_name(path_to_file):
    file_list = []
    for file in os.listdir(path_to_file):
        if file.find(".md") == -1:
            file_list.append(file)
    return file_list


def generate_color_index(path_to_file, file_list, file_format="jpg"):

    color_palette = {}
    index_palette = {}
    for i, file in enumerate(file_list):
        img = Image.open(path_to_file + file)
        img_array = np.asarray(img, dtype=np.uint8)
        rgb_tuple = (img_array[0][0][0], img_array[0][0][1], img_array[0][0][2])

        color_palette.update({rgb_tuple: (i, file.rstrip("." + file_format))})
        index_palette.update({i: (rgb_tuple, file.rstrip("." + file_format))})

    return color_palette, index_palette


if __name__ == "__main__":
    file_list = get_file_name(FILE_PATH)
    color_palette, index_palette = generate_color_index(FILE_PATH, file_list)
    print(file_list)
    print("\n\n")
    print(color_palette)
    print("\n\n")
    print(index_palette)
