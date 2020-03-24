import numpy as np
from PIL import Image
import os
import csv

IMAGE_PATH = "./JPEGImages/"
ANNOTATION_PATH = "./SegmentationClass/"

def generate_color_index(path_to_file, index_dict, file_format="jpg"):

    index_palette = {}
    for i in index_dict.keys():
        img = Image.open(path_to_file + index_dict[i] + "." + file_format)
        img_array = np.asarray(img, dtype=np.uint8)
        rgb_tuple = (img_array[0][0][0], img_array[0][0][1], img_array[0][0][2])

        index_palette.update({i: (rgb_tuple, index_dict[i])})

    return index_palette


def get_file_list_from_directory(directory_name, sort_key=str.lower):
    file_list_buffer = os.listdir(directory_name)
    file_list = [f for f in file_list_buffer if os.path.isfile(directory_name + f)]
    file_list.sort(key=sort_key)
    return file_list, len(file_list)


if __name__ == "__main__":
    print(get_file_list_from_directory(IMAGE_PATH), "\n\n\n\n", get_file_list_from_directory(ANNOTATION_PATH))
