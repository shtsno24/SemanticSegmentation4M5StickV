import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageChops
import Model_V0_1
from tensorflow.keras import Model


try:
    MODEL_FILE = "Model_V0_1.h5"
    LABELS = 5
    COLOR_DEPTH = 3
    CROP_HEIGHT = 32  # sensor.LCD[32, 32]
    CROP_WIDTH = 32

    # Load model
    print("\n\nLoad Model...\n")
    model_old = tf.keras.models.load_model(MODEL_FILE, custom_objects={'loss_function': Model_V0_1.weighted_SparseCategoricalCrossentropy(LABELS)})
    model_new = Model(model_old.input, model_old.layers[-2].output)
    model_new.summary()
    print("\nDone")

    print("\n\nSave Model...")
    model_new.save('Model_V0_1.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")
