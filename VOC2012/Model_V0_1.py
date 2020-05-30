import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU, Flatten


def weighted_SparseCategoricalCrossentropy(sample_weights, classes=5):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.uint8)
        y_true = tf.one_hot(y_true, depth=classes)
        y_true = tf.cast(y_true, tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy()
        error = cce(y_true, y_pred) * sample_weights
        return error
    return loss_function


def DownSampling_block(x, input_depth, channel, Momentum=0.1):
    if channel > input_depth:
        internal_channel = int((channel - input_depth) / 2)
    elif channel < input_depth:
        internal_channel = int((input_depth - channel) / 2)
    else:
        internal_channel = int(channel / 2)

    x_conv = Conv2D(internal_channel, (1, 1))(x)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = Activation("relu")(x_conv)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_3 = DepthwiseConv2D((3, 3))(x_conv)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = Conv2D(internal_channel, (1, 1))(x_conv_3)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_5 = DepthwiseConv2D((5, 5))(x_conv)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)
    x_conv_5 = Conv2D(internal_channel, (1, 1))(x_conv_5)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)

    y = Concatenate(axis=3)([x_conv_3, x_conv_5, x])

    y = Conv2D(channel, (1, 1))(y)
    y = BatchNormalization(momentum=Momentum)(y)
    y = Activation("relu")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    return y


def UpSampling_block(x, input_depth, channel, Momentum=0.1):
    if channel > input_depth:
        internal_channel = int((channel - input_depth) / 2)
    elif channel < input_depth:
        internal_channel = int((input_depth - channel) / 2)
    else:
        internal_channel = int(channel / 2)

    x_conv = Conv2D(internal_channel, (1, 1))(x)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = Activation("relu")(x_conv)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_3 = DepthwiseConv2D((3, 3))(x_conv)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = Conv2D(internal_channel, (1, 1))(x_conv_3)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_5 = DepthwiseConv2D((5, 5))(x_conv)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)
    x_conv_5 = Conv2D(internal_channel, (1, 1))(x_conv_5)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)

    y = Concatenate(axis=3)([x_conv_3, x_conv_5, x])
    y = Conv2D(channel, (1, 1))(y)
    y = BatchNormalization(momentum=Momentum)(y)
    y = Activation("relu")(y)

    y = UpSampling2D(size=(2, 2))(y)

    return y


def Normal_block(x, input_depth, channel, Momentum=0.1):
    if channel > input_depth:
        internal_channel = int((channel - input_depth) / 2)
    elif channel < input_depth:
        internal_channel = int((input_depth - channel) / 2)
    else:
        internal_channel = int(channel / 2)

    x_conv = Conv2D(internal_channel, (1, 1))(x)
    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    x_conv = Activation("relu")(x_conv)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_3 = DepthwiseConv2D((3, 3))(x_conv)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = Conv2D(internal_channel, (1, 1))(x_conv_3)
    x_conv_3 = BatchNormalization(momentum=Momentum)(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)

    x_conv = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_conv)
    x_conv_5 = DepthwiseConv2D((5, 5))(x_conv)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)
    x_conv_5 = Conv2D(internal_channel, (1, 1))(x_conv_5)
    x_conv_5 = BatchNormalization(momentum=Momentum)(x_conv_5)
    x_conv_5 = Activation("relu")(x_conv_5)

    y = Concatenate(axis=3)([x_conv_3, x_conv_5, x])

    y = Conv2D(channel, (1, 1))(y)
    y = BatchNormalization(momentum=Momentum)(y)
    y = Activation("relu")(y)
    return y


def TestNet(input_shape=(32, 32, 3), classes=5):

    input_0 = Input(shape=input_shape)

    x = DownSampling_block(input_0, 3, 32 - 3)
    input_1 = MaxPooling2D(pool_size=(2, 2))(input_0)
    x_1 = Concatenate(axis=3)([x, input_1])

    x_1 = Conv2D(32, (1, 1))(x_1)
    x_1 = BatchNormalization(momentum=0.1)(x_1)
    x_1 = Activation("relu")(x_1)
    x = Normal_block(x_1, 32, 64)

    for _ in range(3 - 1):
        x = Normal_block(x, 64, 64)
    x = DownSampling_block(x, 64, 128 - 3)
    input_2 = MaxPooling2D(pool_size=(2, 2))(input_1)
    x_2 = Concatenate(axis=3)([x, input_2])

    x_2 = Conv2D(64, (1, 1))(x_2)
    x_2 = BatchNormalization(momentum=0.1)(x_2)
    x_2 = Activation("relu")(x_2)
    x = Normal_block(x_2, 64, 128)

    for _ in range(5 - 1):
        x = Normal_block(x, 128, 128)
    x = Normal_block(x, 128, 64)

    x = Concatenate(axis=3)([x, x_2])
    x = UpSampling_block(x, 128, 64)

    x = Normal_block(x, 64, 32)
    x = Concatenate(axis=3)([x, x_1])

    x = UpSampling_block(x, 64, 16)
    x = Normal_block(x, 16, classes)

    outputs = Softmax()(x)

    model = Model(input_0, outputs)
    return model
