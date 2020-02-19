import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, UpSampling2D, Activation, Concatenate
from tensorflow.keras.losses import sparse_categorical_crossentropy

"""
もしかしたら，ResizeNearesetNeighbor，ResizeBilinear, TransposeConvが使えるかも．
H x W x D : 112 x 112 x 3 -> 112 x 112 x (class)
"""


def TestNet(input_shape= (112, 112, 3), classes=151):
    inputs = Input(shape=input_shape)

    x = DepthwiseConv2D((3, 3), padding="same")(inputs)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    # 56 x 56 x 32

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = Conv2D(64, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x1)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 28 x 28 x 64

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x2 = Conv2D(128, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x2)
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 14 x 14 x 128

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, x2])

    # 28 x 28 x 192

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, x1])

    # 56 x 56 x 96

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(classes, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 112 x 112 x 151
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    return model
