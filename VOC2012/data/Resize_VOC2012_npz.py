import os
import csv
import sys
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageChops
from tensorflow.python.client import device_lib


try:
    print(device_lib.list_local_devices())

    # RESIZE_HEIGHT = 128
    # RESIZE_WIDTH = 128
    RESIZE_HEIGHT = 32
    RESIZE_WIDTH = 32
    LABELS = 4  # With BackGround
    COLOR_DEPTH = 3
    SAMPLE = np.random.randint(30)

    TRAIN_RECORDS = "VOC2012_resize_train.npz"
    TRAINVAL_RECORDS = "VOC2012_resize_trainval.npz"
    VAL_RECORDS = "VOC2012_resize_val.npz"

    TRAIN_DATA_NAME = "./ImageSets/Segmentation/train.txt"
    TRAINVAL_DATA_NAME = "./ImageSets/Segmentation/trainval.txt"
    VAL_DATA_NAME = "./ImageSets/Segmentation/val.txt"

    image_file_directories = {"image": "./JPEGImages/", "annotation": "./SegmentationClass/"}
    image_array_list = []
    annotation_array_list = []
    label_balance_array = np.zeros(LABELS)
    label_pixel_count_array = np.zeros(LABELS)
    label_balance_array_resize = np.zeros(LABELS)
    label_pixel_count_array_resize = np.zeros(LABELS)

    # Generate Train Data Records
    try:
        print("\n\n\nGenerate Train Data Records...\n")
        with open(TRAIN_DATA_NAME) as f:
            reader = csv.reader(f)
            show_flag = 0
            data_num = 1463
            for i, name in enumerate(reader):
                # Loading image-like data
                image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                with Image.open(image_file_name) as image_object:
                    image_object = image_object.convert("RGB")
                    image_data = np.array(image_object, dtype=np.uint8)
                    # Detect Edges in image
                    image_edge = image_object.convert("L")
                    image_dilation = image_edge.filter(ImageFilter.MaxFilter(3))
                    image_erosion = image_edge.filter(ImageFilter.MinFilter(3))
                    image_edge = ImageChops.difference(image_dilation, image_erosion)
                    image_edge_array = np.array(image_edge, dtype=np.uint8)
                    # image_data = np.concatenate((image_data, image_edge_array.reshape(image_edge_array.shape + (1,))), axis=2)

                with Image.open(annotation_file_name) as annotation_object:
                    annotation_data = np.array(annotation_object, dtype=np.uint8)
                    annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                    annotation_data[annotation_data == 255] = 0
                    # Reduce index. refer to https://qiita.com/mine820/items/725fe55c095f28bffe87
                    for ic in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 17, 19]:
                        annotation_data[annotation_data == ic] = 0

                    # 0:BG 9:CHAIR 11:TABLE(Same as CHAIR,SOFA), 15:PEOPLE 18:SOFA(Same as CHAIR, TABLE) 20:TV 21:VOID
                    # 0:BG,VOID 1:CHAIR,SOFA,TABLE 3:PEOPLE 4:TV
                    annotation_data[annotation_data == 9] = 1
                    annotation_data[annotation_data == 11] = 1
                    annotation_data[annotation_data == 15] = 2
                    annotation_data[annotation_data == 18] = 1
                    annotation_data[annotation_data == 20] = 3

                    hist, bins = np.histogram(annotation_data, bins=np.arange(LABELS + 1))
                    label_balance_array += hist
                    pixel_cnt = annotation_data.shape[0] * annotation_data.shape[1] * annotation_data.shape[2]
                    label_pixel_count_array[hist > 0] += pixel_cnt
                    if i == 0:
                        palette_raw = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)

                        for index, ic in enumerate([0, 9, 15, 20]):
                            palette_raw[index][0] = palette_raw[ic][0]
                            palette_raw[index][1] = palette_raw[ic][1]
                            palette_raw[index][2] = palette_raw[ic][2]

                        palette = np.copy(palette_raw)
                        # print(palette.shape)

                # Resizeing data
                X, Y = image_data.shape[0], image_data.shape[1]
                if X < Y:
                    long_side = Y
                else:
                    long_side = X
                annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                annotation_data = tf.image.resize(annotation_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                image_data = tf.image.resize(image_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # Reshape annotation_data
                annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])
                annotation_array = annotation_data.numpy()
                image_array = image_data.numpy()

                # Get pixel frequency
                hist, bins = np.histogram(annotation_array, bins=np.arange(LABELS + 1))
                label_balance_array_resize += hist
                pixel_cnt = annotation_array.shape[0] * annotation_array.shape[1]
                label_pixel_count_array_resize[hist > 0] += pixel_cnt

                # Add data to lists
                image_array_list.append(image_array)
                annotation_array_list.append(annotation_array)
                image_array_list.append(image_array[:, ::-1])
                annotation_array_list.append(annotation_array[:, ::-1])

                status = int((i + 1) / data_num * 100)
                if status % 2 == 0:
                    if show_flag == 0:
                        show_flag = 1
                        print("\033[F", int(status), "%|", "■" * int(status / 2))  
                else:
                    show_flag = 0

        annotation_data = annotation_array[:, ::-1]
        annotation_image = Image.fromarray(annotation_data)
        annotation_image.putpalette(palette)
        annotation_image.show()

        image_edge.show()

        annotation_data = np.array(annotation_array_list, dtype=np.uint8)
        image_data = np.array(image_array_list, dtype=np.uint8)
        np.savez(TRAIN_RECORDS,
                 image=image_data,
                 annotation=annotation_data,
                 label_pix_resize=label_balance_array_resize,
                 label_pix=label_balance_array,
                 label_pix_cnt=label_pixel_count_array,
                 label_pix_cnt_resize=label_pixel_count_array_resize)
        print("Labels               :", bins)
        print("Pixels in each label :", label_balance_array)
        print("Pixels in each label(with resized images) :", label_balance_array_resize)
        label_balance_array = np.zeros(LABELS)
        label_balance_array_resize = np.zeros(LABELS)
        label_pixel_count_array = np.zeros(LABELS)
        label_pixel_count_array_resize = np.zeros(LABELS)
        print("\n\nDone")

    except:
        traceback.print_exc()
        sys.exit(1)

    # Generate Train Validate Data Records
    try:
        print("\n\n\nGenerate Train Validate Data Records...\n")
        with open(TRAINVAL_DATA_NAME) as f:
            reader = csv.reader(f)
            show_flag = 0
            data_num = 2912
            for i, name in enumerate(reader):
                # Loading image-like data
                image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                with Image.open(image_file_name) as image_object:
                    image_object = image_object.convert("RGB")
                    image_data = np.array(image_object, dtype=np.uint8)
                    # Detect Edges in image
                    image_edge = image_object.convert("L")
                    image_dilation = image_edge.filter(ImageFilter.MaxFilter(3))
                    image_erosion = image_edge.filter(ImageFilter.MinFilter(3))
                    image_edge = ImageChops.difference(image_dilation, image_erosion)
                    image_edge_array = np.array(image_edge, dtype=np.uint8)
                    # image_data = np.concatenate((image_data, image_edge_array.reshape(image_edge_array.shape + (1,))), axis=2)

                with Image.open(annotation_file_name) as annotation_object:
                    annotation_data = np.array(annotation_object, dtype=np.uint8)
                    annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                    annotation_data[annotation_data == 255] = 0
                    # Reduce index. refer to https://qiita.com/mine820/items/725fe55c095f28bffe87
                    for ic in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 17, 19]:
                        annotation_data[annotation_data == ic] = 0

                    # 0:BG 9:CHAIR 11:TABLE(Same as CHAIR,SOFA), 15:PEOPLE 18:SOFA(Same as CHAIR, TABLE) 20:TV 21:VOID
                    # 0:BG,VOID 1:CHAIR,SOFA,TABLE 3:PEOPLE 4:TV
                    annotation_data[annotation_data == 9] = 1
                    annotation_data[annotation_data == 11] = 1
                    annotation_data[annotation_data == 15] = 2
                    annotation_data[annotation_data == 18] = 1
                    annotation_data[annotation_data == 20] = 3

                    hist, bins = np.histogram(annotation_data, bins=np.arange(LABELS + 1))
                    label_balance_array += hist
                    pixel_cnt = annotation_data.shape[0] * annotation_data.shape[1] * annotation_data.shape[2]
                    label_pixel_count_array[hist > 0] += pixel_cnt
                    if i == 0:
                        palette_raw = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)

                        for index, ic in enumerate([0, 9, 11, 15, 20]):
                            palette_raw[index][0] = palette_raw[ic][0]
                            palette_raw[index][1] = palette_raw[ic][1]
                            palette_raw[index][2] = palette_raw[ic][2]

                        palette = np.copy(palette_raw)

                # Resizeing data
                X, Y = image_data.shape[0], image_data.shape[1]
                if X < Y:
                    long_side = Y
                else:
                    long_side = X
                annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                annotation_data = tf.image.resize(annotation_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                image_data = tf.image.resize(image_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # Reshape annotation_data
                annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])
                annotation_array = annotation_data.numpy()
                image_array = image_data.numpy()

                # Get pixel frequency
                hist, bins = np.histogram(annotation_array, bins=np.arange(LABELS + 1))
                label_balance_array_resize += hist
                pixel_cnt += annotation_array.shape[0] * annotation_array.shape[1]
                label_pixel_count_array_resize[hist > 0] += pixel_cnt

                # Add data to lists
                image_array_list.append(image_array)
                annotation_array_list.append(annotation_array)
                image_array_list.append(image_array[:, ::-1])
                annotation_array_list.append(annotation_array[:, ::-1])

                status = int((i + 1) / data_num * 100)
                if status % 2 == 0:
                    if show_flag == 0:
                        show_flag = 1
                        print("\033[F", int(status), "%|", "■" * int(status / 2))  
                else:
                    show_flag = 0

        annotation_data = annotation_array
        annotation_image = Image.fromarray(annotation_data)
        annotation_image.putpalette(palette)
        annotation_image.show()

        image_edge.show()

        annotation_data = np.array(annotation_array_list, dtype=np.uint8)
        image_data = np.array(image_array_list, dtype=np.uint8)
        np.savez(TRAINVAL_RECORDS,
                 image=image_data,
                 annotation=annotation_data,
                 label_pix_resize=label_balance_array_resize,
                 label_pix=label_balance_array,
                 label_pix_cnt=label_pixel_count_array,
                 label_pix_cnt_resize=label_pixel_count_array_resize)
        print("Labels               :", bins)
        print("Pixels in each label :", label_balance_array)
        print("Pixels in each label(with resized images) :", label_balance_array)
        label_balance_array = np.zeros(LABELS)
        label_balance_array_resize = np.zeros(LABELS)
        label_pixel_count_array = np.zeros(LABELS)
        label_pixel_count_array_resize = np.zeros(LABELS)
        print("\n\nDone")

    except:
        traceback.print_exc()
        sys.exit(2)

    # Generate Validate Data Records
    try:
        print("\n\n\nGenerate Validate Data Records...\n")
        with open(VAL_DATA_NAME) as f:
            reader = csv.reader(f)
            show_flag = 0
            data_num = 1448
            for i, name in enumerate(reader):
                # Loading image-like data
                image_file_name = image_file_directories["image"] + name[0] + ".jpg"
                annotation_file_name = image_file_directories["annotation"] + name[0] + ".png"
                with Image.open(image_file_name) as image_object:
                    image_object = image_object.convert("RGB")
                    image_data = np.array(image_object, dtype=np.uint8)
                    # Detect Edges in image
                    image_edge = image_object.convert("L")
                    image_dilation = image_edge.filter(ImageFilter.MaxFilter(3))
                    image_erosion = image_edge.filter(ImageFilter.MinFilter(3))
                    image_edge = ImageChops.difference(image_dilation, image_erosion)
                    image_edge_array = np.array(image_edge, dtype=np.uint8)
                    # image_data = np.concatenate((image_data, image_edge_array.reshape(image_edge_array.shape + (1,))), axis=2)

                with Image.open(annotation_file_name) as annotation_object:
                    annotation_data = np.array(annotation_object, dtype=np.uint8)
                    annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
                    annotation_data[annotation_data == 255] = 0
                    # Reduce index. refer to https://qiita.com/mine820/items/725fe55c095f28bffe87
                    for ic in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 17, 19]:
                        annotation_data[annotation_data == ic] = 0

                    # 0:BG 9:CHAIR 11:TABLE(Same as CHAIR,SOFA), 15:PEOPLE 18:SOFA(Same as CHAIR, TABLE) 20:TV 21:VOID
                    # 0:BG,VOID 1:CHAIR,SOFA,TABLE 3:PEOPLE 4:TV
                    annotation_data[annotation_data == 9] = 1
                    annotation_data[annotation_data == 11] = 1
                    annotation_data[annotation_data == 15] = 2
                    annotation_data[annotation_data == 18] = 1
                    annotation_data[annotation_data == 20] = 3

                    hist, bins = np.histogram(annotation_data, bins=np.arange(LABELS + 1))
                    label_balance_array += hist
                    pixel_cnt = annotation_data.shape[0] * annotation_data.shape[1] * annotation_data.shape[2]
                    label_pixel_count_array[hist > 0] += pixel_cnt
                    if i == 0:
                        palette_raw = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)

                        for index, ic in enumerate([0, 9, 11, 15, 20]):
                            palette_raw[index][0] = palette_raw[ic][0]
                            palette_raw[index][1] = palette_raw[ic][1]
                            palette_raw[index][2] = palette_raw[ic][2]

                        palette = np.copy(palette_raw)

                # Resizeing data
                X, Y = image_data.shape[0], image_data.shape[1]
                if X < Y:
                    long_side = Y
                else:
                    long_side = X
                annotation_data = tf.image.resize_with_crop_or_pad(annotation_data, long_side, long_side)
                annotation_data = tf.image.resize(annotation_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
                image_data = tf.image.resize(image_data, (RESIZE_HEIGHT, RESIZE_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # Reshape annotation_data
                annotation_data = tf.reshape(annotation_data, annotation_data.shape[:2])
                annotation_array = annotation_data.numpy()
                image_array = image_data.numpy()

                # Get pixel frequency
                hist, bins = np.histogram(annotation_array, bins=np.arange(LABELS + 1))
                label_balance_array_resize += hist
                pixel_cnt += annotation_array.shape[0] * annotation_array.shape[1]
                label_pixel_count_array_resize[hist > 0] += pixel_cnt

                # Add data to lists
                image_array_list.append(image_array)
                annotation_array_list.append(annotation_array)
                image_array_list.append(image_array[:, ::-1])
                annotation_array_list.append(annotation_array[:, ::-1])

                status = int((i + 1) / data_num * 100)
                if status % 2 == 0:
                    if show_flag == 0:
                        show_flag = 1
                        print("\033[F", int(status), "%|", "■" * int(status / 2))  
                else:
                    show_flag = 0

        annotation_data = annotation_array
        annotation_image = Image.fromarray(annotation_data)
        annotation_image.putpalette(palette)
        annotation_image.show()

        image_edge.show()

        annotation_data = np.array(annotation_array_list, dtype=np.uint8)
        image_data = np.array(image_array_list, dtype=np.uint8)
        np.savez(VAL_RECORDS,
                 image=image_data,
                 annotation=annotation_data,
                 label_pix_resize=label_balance_array_resize,
                 label_pix=label_balance_array,
                 label_pix_cnt=label_pixel_count_array,
                 label_pix_cnt_resize=label_pixel_count_array_resize)
        print("Labels               :", bins)
        print("Pixels in each label :", label_balance_array)
        print("Pixels in each label(with resized images) :", label_balance_array)
        label_balance_array = np.zeros(LABELS)
        label_balance_array_resize = np.zeros(LABELS)
        label_pixel_count_array = np.zeros(LABELS)
        label_pixel_count_array_resize = np.zeros(LABELS)
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
