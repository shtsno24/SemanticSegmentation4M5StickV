import numpy as np


def create_scene_parse150_label_colormap():
    """
    Creates a label colormap used in Scene Parse150 segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.

    https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py
    """

    colormap_array = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230], #
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7], #
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92], # 
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255], #
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255], #
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ], dtype=np.uint8)

    colormap_dict = {0 : (colormap_array[0][0], colormap_array[0][1], colormap_array[0][2])}
    for i in range(colormap_array.shape[0]):
        colormap_dict.update({i : (colormap_array[i][0], colormap_array[i][1], colormap_array[i][2])})

    return colormap_dict

def rgb2onehot(x, DTYPE=np.uint8):
    color_map = create_scene_parse150_label_colormap()
    classes = len(color_map)
    shapes = x.shape[:2] + (classes,)

    output_array = np.zeros((shapes), dtype=np.uint8)
    for i, buff in enumerate(color_map):
        output_array[:,:,i] = np.all(x.reshape((-1,3)) == color_map[i], axis=1).reshape(shapes[:2])

    return output_array.astype(DTYPE)


def test():
    array_size = 5
    array_depth = 3
    x = np.zeros((array_size, array_size, array_depth), dtype=np.uint8)
    index_map = np.zeros((array_size, array_size), dtype=np.uint8)
    color_map = create_scene_parse150_label_colormap()
    classes = len(color_map)

    for X in range(array_size):
        for Y in range(array_size):
            rand_index = np.random.randint(0, classes)
            index_map[X][Y] = rand_index
            for i in range(array_depth):
                x[X][Y][i] = color_map[rand_index][i]

    shapes = x.shape[:2] + (classes,)
    output_array = np.zeros((shapes), dtype=np.uint8)
    for i, buff in enumerate(color_map):
        output_array[:,:,i] = np.all(x.reshape((-1,3)) == color_map[i], axis=1).reshape(shapes[:2])

    print(np.array_equal(output_array.argmax(axis=2).astype(np.uint8), index_map))
    print("\n\n")
    print(output_array.argmax(axis=2).astype(np.uint8))
    print("\n\n")
    print(index_map)
    return output_array

if __name__ == "__main__":
    test()