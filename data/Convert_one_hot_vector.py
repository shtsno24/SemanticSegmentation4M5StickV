import numpy as np
import time
from get_color_label import get_file_name, generate_color_index

FILE_PATH = "./color150/"


def create_scene_parse150_label_colormap(path_to_file, file_format="jpg"):
    """
    Creates a label colormap and label names used in Scene Parse150 segmentation benchmark.
    https://github.com/CSAILVision/sceneparsing/tree/master/visualizationCode/color150
    """
    file_list = get_file_name(path_to_file)
    color_dict, index_dict = generate_color_index(path_to_file, file_list, file_format=file_format)

    return color_dict, index_dict


def test_create_label():
    color_dict, index_dict = create_scene_parse150_label_colormap()
    print(len(color_dict), len(index_dict))
    print("\n\n")
    print(color_dict)
    print("\n\n")
    print(index_dict)


def rgb2onehot(x, Color_map, DTYPE=np.uint8):
    color_map = Color_map
    classes = len(color_map)
    shapes = x.shape[:2] + (classes,)

    # output_array = np.zeros((shapes), dtype=np.uint8)
    # for i, buff in enumerate(color_map):
    #     output_array[:, :, i] = np.all(x.reshape((-1, 3)) == color_map[i], axis=1).reshape(shapes[:2])
    output_array = np.zeros(shapes, dtype=DTYPE)
    i = 0
    for X in range(output_array.shape[0]):
        for Y in range(output_array.shape[1]):
            try:
                color_tuple = (x[X][Y][0], x[X][Y][1], x[X][Y][2])
                output_array[X][Y][color_map[color_tuple][0]] = 1
            except KeyError:
                i += 1
                print(i, "KeyError :", color_tuple)
                output_array[X][Y][0] = 1

    return output_array.astype(DTYPE)


def test(Color_map, Index_map):
    array_size = 128
    array_depth = 3
    x = np.zeros((array_size, array_size, array_depth), dtype=np.uint8)
    index_map = np.zeros((array_size, array_size), dtype=np.uint8)
    color_map = Color_map
    Index_map = Index_map
    classes = len(color_map)
    print(classes)
    print("\n\n")
    i = 0
    for X in range(array_size):
        for Y in range(array_size):
            rand_index = np.random.randint(0, classes)
            index_map[X][Y] = rand_index
            buff_color_list = Index_map[rand_index][0]
            for i in range(array_depth):
                x[X][Y][i] = buff_color_list[i]

    # shapes = x.shape[:2] + (classes,)
    # output_array = np.zeros((shapes), dtype=np.uint8)
    # for i, buff in enumerate(color_map):
    #     output_array[:, :, i] = np.all(x.reshape((-1, 3)) == color_map[i], axis=1).reshape(shapes[:2])

    output_array = np.zeros((x.shape[:2]), dtype=np.uint8)
    for X in range(array_size):
        for Y in range(array_size):
            color_tuple = (x[X][Y][0], x[X][Y][1], x[X][Y][2])
            output_array[X][Y] = color_map[color_tuple][0]

    print(np.array_equal(output_array, index_map))
    print("\n\n")
    print(output_array)
    print("\n\n")
    print(index_map)
    return output_array


if __name__ == "__main__":
    start = time.time()
    color_map, index_map = create_scene_parse150_label_colormap(FILE_PATH)
    for i in range(1):
        test(color_map, index_map)
        # test_create_label()
    end = time.time()
    print(str(end - start) + "[s]")
