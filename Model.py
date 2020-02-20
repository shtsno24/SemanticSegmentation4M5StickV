import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, UpSampling2D, Activation, Concatenate
from tensorflow.keras.losses import sparse_categorical_crossentropy

"""
もしかしたら，ResizeNearesetNeighbor，ResizeBilinear, TransposeConvが使えるかも．
H x W x D : 112 x 112 x 3 -> 112 x 112 x (class)
"""


"""
def TestNet(input_shape= (112, 112, 3), classes=151):
    inputs = Input(shape=input_shape)

    x = DepthwiseConv2D((3, 3), padding="same")(inputs)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x1 = MaxPooling2D(pool_size=(4, 4))(x)

    # 56 x 56 x 32

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x2 = MaxPooling2D(pool_size=(2, 2))(x)

    # 28 x 28 x 64

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 14 x 14 x 128

    x = Concatenate()([x, x1, x2])
    x = Conv2D(128, (1, 1), activation="relu")(x)

    # 14 x 14 x 224

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 28 x 28 x 64

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 56 x 56 x 32

    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(classes, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 112 x 112 x 151
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    return model
"""


def TestNet(input_shape= (112, 112, 3), classes=151):
    inputs = Input(shape=input_shape)

    x0 = DepthwiseConv2D((3, 3), padding="same")(inputs)
    x1 = DepthwiseConv2D((5, 5), padding="same")(inputs)
    x2 = DepthwiseConv2D((7, 7), padding="same")(inputs)
    x3 = DepthwiseConv2D((9, 9), padding="same")(inputs)
    x = Concatenate()([x0, x1, x2, x3])
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 56 x 56 x 64

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 28 x 28 x 128

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(256, (1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 14 x 14 x 256

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(256, (1, 1), activation="relu")(x)

    # 14 x 14 x 256

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 28 x 28 x 128

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 56 x 56 x 64

    x0 = DepthwiseConv2D((3, 3), padding="same")(x)
    x1 = DepthwiseConv2D((7, 7), padding="same")(x)
    x = Concatenate()([x0, x1])
    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 112 x 112 x 151
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = Conv2D(classes, (1, 1), activation="relu")(x)
    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    return model


def TestNet2(input_shape=(112, 112, 3), classes=151):
    inputs = Input(shape=input_shape)

    x = MaxPooling2D(pool_size=(2, 2))(inputs)
    x_3 = DepthwiseConv2D((3, 3), padding="same")(x)
    x_5 = DepthwiseConv2D((5, 5), padding="same")(x)
    x_7 = DepthwiseConv2D((7, 7), padding="same")(x)
    x_9 = DepthwiseConv2D((9, 9), padding="same")(x)
    x = Concatenate()([x, x_3, x_5, x_7, x_9])
    x = Conv2D(64, (1, 1), activation="relu")(x)

    # 56 x 56 x 64

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(64, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(64, (1, 1), activation="relu")(x_7)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(128, (1, 1), activation="relu")(x)

    # 28 x 28 x 128

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(128, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(128, (1, 1), activation="relu")(x_3)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(256, (1, 1), activation="relu")(x)

    # 14 x 14 x 256

    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(256, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(256, (1, 1), activation="relu")(x_3)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(256, (1, 1), activation="relu")(x)

    # 14 x 14 x 256

    x = UpSampling2D(size=(2, 2))(x)
    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(256, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(256, (1, 1), activation="relu")(x_3)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(128, (1, 1), activation="relu")(x)

    # 28 x 28 x 128

    x = UpSampling2D(size=(2, 2))(x)
    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(128, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(128, (1, 1), activation="relu")(x_3)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(64, (1, 1), activation="relu")(x)

    # 56 x 56 x 64

    x = UpSampling2D(size=(2, 2))(x)
    x_3 = DepthwiseConv2D((3, 1), padding="same", activation="relu")(x)
    x_3 = DepthwiseConv2D((1, 3), padding="same", activation="relu")(x_3)
    x_3 = Conv2D(64, (1, 1), activation="relu")(x_3)
    x_7 = DepthwiseConv2D((7, 1), padding="same", activation="relu")(x)
    x_7 = DepthwiseConv2D((1, 7), padding="same", activation="relu")(x_7)
    x_7 = Conv2D(64, (1, 1), activation="relu")(x_3)
    x = Concatenate()([x, x_3, x_7])
    x = Conv2D(classes, (1, 1), activation="relu")(x)

    # 112 x 112 x classes

    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    return model
