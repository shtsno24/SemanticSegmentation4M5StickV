import numpy as np
import tensorflow as tf
from tensorflow import keras

import Model

try:
    PATH = "data/scene_parse150_resize.npz"
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 100
    EPOCHS = 10

    # Load data from .npz
    print("\n\nLoad data...", end="")
    with np.load(PATH) as data:
        train_image = data["train_image"]
        train_annotation = data["train_annotation"]
        test_image = data["test_image"]
        test_annotation = data["test_annotation"]
    print("  Done\n\n")

    # Generate dataset
    print("Generate dataset...", end="")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_annotation))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_annotation))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    print("  Done\n\n")

    # Define model
    print("Load Model...\n\n")
    model = Model.TestNet()
    model.summary()
    print("\n\nDone\n\n")

    # Train model
    print("Train Model...", end="")
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    # model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)
    # model.save('TestNet.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    input(">")