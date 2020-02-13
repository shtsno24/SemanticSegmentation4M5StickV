import os
import numpy as np
import tensorflow as tf
from PIL import Image
from Util import label2onehot, create_scene_parse150_label_dict

try:
    CROP_HEIGHT = 112
    CROP_WIDTH = 112
    LABELS = 151
    SAMPLE = np.random.randint(30)
    # SAMPLE = 0
    FILE_PATH = "./color150/"
    OBJECT_FILE = "objectinfo150.csv"
    INDEX_PALETTE = create_scene_parse150_label_dict(FILE_PATH, OBJECT_FILE)

    dataset_lists = {"train_image": [], "train_annotation": [], "test_image": [], "test_annotation": []}
    image_file_directories = {"train_image": "./ADEChallengeData2016/images/training/",
                              "train_annotation": "./ADEChallengeData2016/annotations/training/",
                              "test_image": "./ADEChallengeData2016/images/validation/",
                              "test_annotation": "./ADEChallengeData2016/annotations/validation/"}

    # Process image
    print("\n\n\nResizing Images...")
    for Mode in ("train_image", "train_annotation", "test_image", "test_annotation"):
    # for Mode in ("test_image", "test_annotation"):
        file_list_buffer = os.listdir(image_file_directories[Mode])
        file_list = [f for f in file_list_buffer if os.path.isfile(image_file_directories[Mode] + f)]
        Data_num = len(file_list)
        print(Mode, Data_num)
        i = 0
        for f_name in file_list:
            raw_data = Image.open(image_file_directories[Mode] + f_name)
            image_data = np.array(raw_data, dtype=np.uint8)
            if i == SAMPLE:
                pil_img = Image.fromarray(image_data)
                pil_img.save(Mode + ".png")
            X, Y = image_data.shape[0], image_data.shape[1]
            if X < Y:
                long_side = Y
            else:
                long_side = X

            if image_data.ndim == 2:
                image_data = image_data.reshape(image_data.shape + (1,))

            image_data = tf.image.resize_with_crop_or_pad(image_data, long_side, long_side)
            image_data = tf.image.resize(image_data, (CROP_HEIGHT, CROP_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            image_data = image_data.numpy()
            if Mode.find("annotation") != -1:
                image_data = image_data.reshape(image_data.shape[:2])
                output_data = label2onehot(image_data, axis=2, DTYPE=image_data.dtype, labels=LABELS)
            else:
                output_data = image_data

            if i == SAMPLE:
                if output_data.shape[2] > 3:
                    output_data_argmax = output_data.argmax(axis=2).astype(np.uint8)
                    pil_img = Image.fromarray(output_data_argmax)
                else:
                    pil_img = Image.fromarray(output_data)
                pil_img.save(Mode + "_resized.png")
            i += 1
            dataset_lists[Mode].append(output_data)
        print("Done")

    print("\n\n\n")
    print("train_image :", len(dataset_lists["train_image"]), dataset_lists["train_image"][0].shape, type(dataset_lists["train_image"][0]))
    print("train_annotation :", len(dataset_lists["train_annotation"]), dataset_lists["train_annotation"][0].shape, type(dataset_lists["train_annotation"][0]))
    print("test_image :", len(dataset_lists["test_image"]), dataset_lists["test_image"][0].shape, type(dataset_lists["test_image"][0]))
    print("test_annotation :", len(dataset_lists["test_annotation"]), dataset_lists["test_annotation"][0].shape, type(dataset_lists["test_annotation"][0]))

    print("\n\n\nConvert to ndarray...   ", end="")

    train_image_array = np.asarray(dataset_lists["train_image"], dtype=np.uint8)
    train_annotation_array = np.asarray(dataset_lists["train_annotation"], dtype=np.uint8)
    test_image_array = np.asarray(dataset_lists["test_image"], dtype=np.uint8)
    test_annotation_array = np.asarray(dataset_lists["test_annotation"], dtype=np.uint8)

    # shuffle data
    train_p = np.random.permutation(train_image_array.shape[0])
    train_image_array = train_image_array[train_p]
    train_annotation_array = train_annotation_array[train_p]

    test_p = np.random.permutation(test_image_array.shape[0])
    test_image_array = test_image_array[test_p]
    test_annotation_array = test_annotation_array[test_p]



    print("Done")
    print("\n\n\nConpressed data...   ", end="")
    np.savez_compressed("scene_parse150_resize",
                        train_image=train_image_array,
                        train_annotation=train_annotation_array,
                        test_image=test_image_array,
                        test_annotation=test_annotation_array)

    print("Saved")

except:
    import traceback
    traceback.print_exc()


finally:
    print(">>")
