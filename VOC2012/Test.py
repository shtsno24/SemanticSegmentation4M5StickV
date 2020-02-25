import numpy as np
import tensorflow as tf
from PIL import Image


try:

    TEST_IMAGE = "./data/JPEGImages/2007_000346.jpg"
    TEST_ANNOTATION = "./data/SegmentationClass/2007_000346.png"
    MODEL_FILE = "TestNet_VOC2012_npz.h5"
    LABELS = 21
    COLOR_DEPTH = 3
    CROP_HEIGHT = 120
    CROP_WIDTH = 160

    # Load data
    print("\n\nLoad data...\n")
    with Image.open(TEST_ANNOTATION) as annotation_object:
        annotation_data = np.array(annotation_object, dtype=np.uint8)
        annotation_data = annotation_data.reshape(annotation_data.shape + (1,))
        annotation_data[annotation_data == 255] = 0
        palette = np.array(annotation_object.getpalette(), dtype=np.uint8).reshape(-1, 3)
        palette[21] = 255
    with Image.open(TEST_IMAGE) as image_object:
        image_object = image_object.convert("RGB")
        image_data = np.array(image_object, dtype=np.uint8)
        image_data = image_data.reshape((1,) + image_data.shape)

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

    print(annotation_data.shape, image_data.shape, "\nDone")

    # Load model
    print("\n\nLoad Model...\n")
    model = tf.keras.models.load_model(MODEL_FILE)
    model.summary()
    print("\nDone")

    # Prediction
    print("\n\nPrediction...\n")
    image_data = tf.cast(image_data, tf.float32) / 255.0
    prediction_data = model.predict(image_data)
    prediction_data = tf.reshape(prediction_data, [CROP_HEIGHT, CROP_WIDTH, LABELS])
    prediction_data = tf.argmax(prediction_data, axis=2)
    image_data = tf.cast(image_data, tf.float32) * 255.0
    print(prediction_data.shape, "\nDone")

    # Show Prediction
    print("\n\nSave Image...\n")
    prediction_object = Image.fromarray(prediction_data.numpy().astype(np.uint8))
    prediction_object.putpalette(palette)
    prediction_object.save("Test_Prediction.png")
    annotation_object = Image.fromarray(annotation_data.numpy().reshape((CROP_HEIGHT, CROP_WIDTH)).astype(np.uint8))
    annotation_object.putpalette(palette)
    annotation_object.save("Test_Annotation.png")
    image_object = Image.fromarray(image_data.numpy().reshape((CROP_HEIGHT, CROP_WIDTH, COLOR_DEPTH)).astype(np.uint8))
    image_object.save("Test_Image.png")
    print("\nDone")
except:
    import traceback
    traceback.print_exc()

finally:
    input(">")