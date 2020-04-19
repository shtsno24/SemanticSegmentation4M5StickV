import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy


def weighted_SparseCategoricalCrossentropy(sample_weights, classes=5):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.uint8)
        y_true = tf.one_hot(y_true, depth=classes)
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true * sample_weights
        # cce = tf.keras.losses.SparseCategoricalCrossentropy()
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(y_true, y_pred)
    return loss_function


def SkipConvBlock(x, output_channel, kernel_size=(3, 3), padding=((1, 1), (1, 1)), activation=True, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag)

    Conv = Conv2D(internal_channel, (1, 1))(x)

    Conv = ZeroPadding2D(padding=padding)(Conv)
    Conv = DepthwiseConv2D(kernel_size)(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    Conv = Conv2D(output_channel, (1, 1))(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    x = Concatenate(axis=3)([Conv, x])
    x = Conv2D(output_channel, (1, 1))(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = SpatialDropout2D(drop_rate)(x)
    if activation is True:
        x = ReLU()(x)

    return x


def SkipConvBlockD(x, output_channel, kernel_size=(3, 3), padding=((1, 1), (1, 1)), activation=True, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag)

    Conv = Conv2D(internal_channel, (1, 1))(x)

    Conv = ZeroPadding2D(padding=padding)(Conv)
    Conv = DepthwiseConv2D(kernel_size)(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    Conv = Conv2D(output_channel, (1, 1))(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    x = Concatenate(axis=3)([x, Conv])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(output_channel, (1, 1))(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = SpatialDropout2D(drop_rate)(x)
    if activation is True:
        x = ReLU()(x)

    return x


def SkipConvBlockU(x, output_channel, kernel_size=(3, 3), padding=((1, 1), (1, 1)), activation=True, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag)

    Conv = Conv2D(internal_channel, (1, 1))(x)

    Conv = ZeroPadding2D(padding=padding)(Conv)
    Conv = DepthwiseConv2D(kernel_size)(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    Conv = Conv2D(output_channel, (1, 1))(Conv)
    Conv = BatchNormalization(momentum=momentum)(Conv)
    Conv = ReLU()(Conv)

    x = Concatenate(axis=3)([x, Conv])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(output_channel, (1, 1))(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = SpatialDropout2D(drop_rate)(x)
    if activation is True:
        x = ReLU()(x)

    return x


def TestNet(input_shape=(32, 32, 3), classes=5):
    inputs = Input(shape=input_shape)

    x0 = SkipConvBlockD(inputs, 16, internal_mag=2)
    for _ in range(2):
        x0 = SkipConvBlock(x0, 16, internal_mag=2)

    x1 = SkipConvBlockD(x0, 32)
    for _ in range(4):
        x1 = SkipConvBlock(x1, 32)

    x2 = SkipConvBlockD(x1, 64)
    for _ in range(16):
        x2 = SkipConvBlock(x2, 64)

    x2 = UpSampling2D(size=(4, 4))(x2)
    x1 = UpSampling2D(size=(2, 2))(x1)
    x = Concatenate(axis=3)([x0, x1, x2])

    for _ in range(2):
        x = SkipConvBlock(x, 32)

    x = SkipConvBlockU(x, classes, internal_mag=2, activation=False)
    x = Reshape((32*32, classes))(x)
    x = Softmax()(x)
    x = Reshape((32, 32, classes))(x)
    outputs = x
    model = Model(inputs, outputs)
    return model
