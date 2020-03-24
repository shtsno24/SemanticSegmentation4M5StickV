import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D, Softmax, ReLU
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


def Multiscale_Depthwise_Conv(x, out_depth, internal_scale=4, Momentum=0.1, Droprate=0.01, Alpha=0.1, Activation_Func=True):
    internal_depth = int(out_depth / internal_scale)

    x = Conv2D(internal_depth, (1, 1))(x)
    x = ReLU()(x)

    x_3 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x_3 = DepthwiseConv2D((3, 3))(x_3)
    x_3 = BatchNormalization(momentum=Momentum)(x_3)
    x_3 = ReLU()(x_3)

    x_5 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_3)
    x_5 = DepthwiseConv2D((3, 3))(x_5)
    x_5 = BatchNormalization(momentum=Momentum)(x_5)
    x_5 = ReLU()(x_5)

    x_7 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_5)
    x_7 = DepthwiseConv2D((3, 3))(x_7)
    x_7 = BatchNormalization(momentum=Momentum)(x_7)
    x_7 = ReLU()(x_7)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = Concatenate(axis=3)([x, x_3, x_5, x_7])
    x = Conv2D(out_depth, (1, 1))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(Droprate)(x)
    if Activation_Func == True:
        x = ReLU()(x)

    return x


def Multiscale_Depthwise_Conv_Downsize(x, out_depth, internal_scale=4, Momentum=0.1, Droprate=0.01, Alpha=0.1, Activation_Func=True):
    internal_depth = int(out_depth / internal_scale)

    x = Conv2D(internal_depth, (1, 1))(x)
    x = ReLU()(x)

    x_3 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x_3 = DepthwiseConv2D((3, 3))(x_3)
    x_3 = BatchNormalization(momentum=Momentum)(x_3)
    x_3 = ReLU()(x_3)

    x_5 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_3)
    x_5 = DepthwiseConv2D((3, 3))(x_5)
    x_5 = BatchNormalization(momentum=Momentum)(x_5)
    x_5 = ReLU()(x_5)

    x_7 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_5)
    x_7 = DepthwiseConv2D((3, 3))(x_7)
    x_7 = BatchNormalization(momentum=Momentum)(x_7)
    x_7 = ReLU()(x_7)

    x = Concatenate(axis=3)([x, x_3, x_5, x_7])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(out_depth, (1, 1))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(Droprate)(x)
    x = ReLU()(x)

    return x


def Multiscale_Depthwise_Conv_Upsize(x, out_depth, internal_scale=4, Momentum=0.1, Droprate=0.01, Alpha=0.1, Activation_Func=True):
    internal_depth = int(out_depth / internal_scale)

    x = Conv2D(internal_depth, (1, 1))(x)
    x = ReLU()(x)

    x_3 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x_3 = DepthwiseConv2D((3, 3))(x_3)
    x_3 = BatchNormalization(momentum=Momentum)(x_3)
    x_3 = ReLU()(x_3)

    x_5 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_3)
    x_5 = DepthwiseConv2D((3, 3))(x_5)
    x_5 = BatchNormalization(momentum=Momentum)(x_5)
    x_5 = ReLU()(x_5)

    x_7 = ZeroPadding2D(padding=((1, 1), (1, 1)))(x_5)
    x_7 = DepthwiseConv2D((3, 3))(x_7)
    x_7 = BatchNormalization(momentum=Momentum)(x_7)
    x_7 = ReLU()(x_7)

    x = Concatenate(axis=3)([x, x_3, x_5, x_7])
    x = UpSampling2D(size=(2, 2))(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = DepthwiseConv2D((3, 3))(x)
    x = Conv2D(out_depth, (1, 1))(x)
    x = BatchNormalization(momentum=Momentum)(x)
    x = SpatialDropout2D(Droprate)(x)
    x = ReLU()(x)

    return x


"""
def TestNet(input_shape=(128, 160, 3), classes=21):
    inputs = Input(shape=input_shape)

    x = Multiscale_Depthwise_Conv_Downsize(inputs, 32)
    x_0 = x
    # 64 x 80 x 32

    for _ in range(2):
        x = Multiscale_Depthwise_Conv(x, 64)
    x = Multiscale_Depthwise_Conv_Downsize(x, 128)
    x_1 = x
    # 32 x 40 x 128

    for _ in range(2):
        x = Multiscale_Depthwise_Conv(x, 128, internal_scale=8)
    x = Multiscale_Depthwise_Conv_Downsize(x, 256)
    x_2 = x
    # 16 x 20 x 256

    for _ in range(1):
        x = Multiscale_Depthwise_Conv(x, 256, internal_scale=8)
    x = Multiscale_Depthwise_Conv_Downsize(x, 256)
    x_3 = x
    # 8 x 10 x 256

    for _ in range(1):
        x = Multiscale_Depthwise_Conv(x, 256, internal_scale=8)
    x = Multiscale_Depthwise_Conv_Downsize(x, 256)
    x_4 = x
    # 4 x 5 x 512

    x = Multiscale_Concat(x_0, x_1, x_2, x_3, x_4, 256)

    for _ in range(2):
        x = Multiscale_Depthwise_Conv(x, 256, internal_scale=8)

    x = Concatenate(axis=3)([x, x_2])
    x = Multiscale_Depthwise_Conv_Upsize(x, 128)

    # 32 x 40 x 128
    for _ in range(1):
        x = Multiscale_Depthwise_Conv(x, 128)

    x = Concatenate(axis=3)([x, x_1])
    x = Multiscale_Depthwise_Conv_Upsize(x, 64)
    # 64 x 80 x 64
    for _ in range(1):
        x = Multiscale_Depthwise_Conv(x, 64)

    x = Concatenate(axis=3)([x, x_0])
    x = Multiscale_Depthwise_Conv_Upsize(x, 32)
    # 128 x 160 x 32
    x = Multiscale_Depthwise_Conv(x, classes)

    # 128 x 160 x classes
    outputs = Softmax()(x)

    model = Model(inputs, outputs)
    return model
"""


def TestNet(input_shape=(128, 128, 3), classes=5):
    inputs = Input(shape=input_shape)

    # x = MaxPooling2D(pool_size=(4, 4))(inputs)

    x_2 = Multiscale_Depthwise_Conv_Downsize(inputs, 16)
    x_2 = Multiscale_Depthwise_Conv(x_2, 16)

    x_4 = Multiscale_Depthwise_Conv_Downsize(x_2, 32)
    x_4 = Multiscale_Depthwise_Conv(x_4, 32)
    x_4 = Multiscale_Depthwise_Conv(x_4, 32)

    x_8 = Multiscale_Depthwise_Conv_Downsize(x_4, 64)
    x_8 = Multiscale_Depthwise_Conv(x_8, 64)
    x_8 = Multiscale_Depthwise_Conv(x_8, 64)
    # x_8 = Multiscale_Depthwise_Conv(x_8, 64)

    x_16 = Multiscale_Depthwise_Conv_Downsize(x_8, 64)
    x_16 = Multiscale_Depthwise_Conv(x_16, 64)
    x_16 = Multiscale_Depthwise_Conv(x_16, 64)
    # x_16 = Multiscale_Depthwise_Conv(x_16, 128)

    # x_2 = Conv2D(16, (1, 1))(x_2)
    # x_2 = BatchNormalization(momentum=0.1)(x_2)
    # x_2 = ReLU()(x_2)
    # x_2 = MaxPooling2D(pool_size=(2, 2))(x_2)

    x_4 = Conv2D(16, (1, 1))(x_4)
    x_4 = BatchNormalization(momentum=0.1)(x_4)
    x_4 = ReLU()(x_4)
    x_4 = UpSampling2D(size=(2, 2))(x_4)

    x_8 = Conv2D(16, (1, 1))(x_8)
    x_8 = BatchNormalization(momentum=0.1)(x_8)
    x_8 = ReLU()(x_8)
    x_8 = UpSampling2D(size=(4, 4))(x_8)

    x_16 = Conv2D(16, (1, 1))(x_16)
    x_16 = BatchNormalization(momentum=0.1)(x_16)
    x_16 = ReLU()(x_16)
    x_16 = UpSampling2D(size=(8, 8))(x_16)

    x = Concatenate(axis=3)([x_2, x_4, x_8, x_16])
    x = Multiscale_Depthwise_Conv(x, 32)
    x = Multiscale_Depthwise_Conv(x, 16)

    x = UpSampling2D(size=(2, 2))(x)
    x = Multiscale_Depthwise_Conv(x, classes, Activation_Func=False)

    outputs = Softmax()(x)
    # outputs = UpSampling2D(size=(2, 2), interpolation='bilinear')(outputs)

    model = Model(inputs, outputs)
    return model

    # 2097152 B
