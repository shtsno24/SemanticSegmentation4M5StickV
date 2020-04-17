import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy


def weighted_SparseCategoricalCrossentropy(sample_weights, classes=4):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.uint8)
        y_true = tf.one_hot(y_true, depth=classes)
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true * sample_weights
        # cce = tf.keras.losses.SparseCategoricalCrossentropy()
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(y_true, y_pred)
    return loss_function


def SkipConvBlock(x, output_channel, kernel_size, padding, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag) * (internal_mag - 1)
    skip_channel = output_channel - internal_channel

    C = Conv2D(internal_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(x)
    C = ReLU()(C)
    C = ZeroPadding2D(padding=padding)(C)
    C = DepthwiseConv2D(kernel_size)(C)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)
    C = Conv2D(output_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)

    Skip = Conv2D(skip_channel, (1, 1))(x)
    Skip = BatchNormalization(momentum=momentum)(Skip)

    x = Concatenate(axis=3)([Skip, C])
    x = Conv2D(output_channel, (1, 1))(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = ReLU()(x)

    return x


def SkipConvBlockD(x, output_channel, kernel_size, padding, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag) * (internal_mag - 1)
    skip_channel = output_channel - internal_channel

    C = Conv2D(internal_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(x)
    C = ReLU()(C)
    C = ZeroPadding2D(padding=padding)(C)
    C = DepthwiseConv2D(kernel_size)(C)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)
    C = Conv2D(output_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)

    Skip = Conv2D(skip_channel, (1, 1))(x)
    Skip = BatchNormalization(momentum=momentum)(Skip)

    x = Concatenate(axis=3)([Skip, C])
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = ReLU()(x)

    return x


def SkipConvBlockU(x, output_channel, kernel_size, padding, activation=True, internal_mag=4, momentum=0.1, drop_rate=0.01):
    internal_channel = int(output_channel / internal_mag) * (internal_mag - 1)
    skip_channel = output_channel - internal_channel

    C = Conv2D(internal_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(x)
    C = ReLU()(C)
    C = ZeroPadding2D(padding=padding)(C)
    C = DepthwiseConv2D(kernel_size)(C)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)
    C = Conv2D(output_channel, (1, 1))(x)
    C = BatchNormalization(momentum=momentum)(C)
    C = ReLU()(C)

    Skip = Conv2D(skip_channel, (1, 1))(x)
    Skip = BatchNormalization(momentum=momentum)(Skip)

    x = Concatenate(axis=3)([Skip, C])
    x = Conv2DTranspose(output_channel, (2, 2), strides=(2, 2))(x)
    x = BatchNormalization(momentum=momentum)(x)
    if activation is True:
        x = ReLU()(x)

    return x


def TestNet(input_shape=(64, 64, 3), classes=5):
    inputs = Input(shape=input_shape)

    x0 = SkipConvBlockD(inputs, 32, (3, 3), padding=(1, 1), internal_mag=2)
    for _ in range(2):
        x0 = SkipConvBlock(x0, 32, (3, 3), padding=(1, 1))
    # 32 x 32 x 32
    x1 = SkipConvBlockD(x0, 64, (3, 3), padding=(1, 1))
    for _ in range(4):
        x1 = SkipConvBlock(x1, 64, (3, 3), padding=(1, 1))
    # 16 x 16 x 64
    x2 = SkipConvBlockD(x1, 128, (3, 3), padding=(1, 1))
    for _ in range(8):
        x2 = SkipConvBlock(x2, 128, (3, 3), padding=(1, 1))
    # 8 x 8 x 128

    x1 = UpSampling2D(size=(2, 2))(x1)
    x2 = UpSampling2D(size=(4, 4))(x2)
    x = Concatenate(axis=3)([x0, x1, x2])
    # 32 x 32 x (32 + 64 + 128)
    x = SkipConvBlockU(x, classes, (3, 3), padding=(1, 1), activation=False)

    outputs = Activation("softmax")(x)
    model = Model(inputs, outputs)
    return model