import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers.core import Activation

"""
もしかしたら，ResizeNearesetNeighbor，ResizeBilinear, TransposeConvが使えるかも．
H x W x D : 120 x 160 x 3 -> 120 x 160 x (class)
"""


def TestNet(input_shape=(120, 160, 3), classes=150):
    inputs = Input(shape=(120, 160, 3))

    x = DepthwiseConv2D(3, (3, 3), padding="same")(inputs)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))

    # 60 x 80 x 32

    x = DepthwiseConv2D(32, (3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))

    # 30 x 40 x 64

    x = DepthwiseConv2D(64, (3, 3), padding="same")(x)
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))

    # 15 x 20 x 128

    x = DepthwiseConv2D(128, (3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 30 x 40 x 64

    x = DepthwiseConv2D(64, (3, 3), padding="same")(x)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 60 x 80 x 32

    x = DepthwiseConv2D(32, (3, 3), padding="same")(x)
    x = Conv2D(classes, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 120 x 160 x 150

    x = DepthwiseConv2D(classes, (3, 3), padding="same")(x)
    x = Activation("softmax")(x)

    model = Model(input=inputs, output=x)
    return model
