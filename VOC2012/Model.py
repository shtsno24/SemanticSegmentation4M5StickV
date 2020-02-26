import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.losses import sparse_categorical_crossentropy

"""
もしかしたら，ResizeNearesetNeighbor，ResizeBilinear, TransposeConvが使えるかも．
H x W x D : 120 x 160 x 3 -> 120 x 160 x (class)
"""


def weighted_SparseCategoricalCrossentropy(sample_weights, classes):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.uint8)
        y_true = tf.one_hot(y_true, depth=classes)
        y_true = tf.cast(y_true, tf.float32)
        loss = y_true * tf.math.log(y_pred) * sample_weights
        return -1 * tf.math.reduce_mean(loss)
    return loss_function


def initial_block(x, channel, kernel_size=(3, 3), stride=(2, 2)):
    x_conv = Conv2D(61, kernel_size, padding="same", strides=stride)(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)

    x_pool = MaxPooling2D(pool_size=(2, 2))(x)
    y = Concatenate(axis=3)([x_conv, x_pool])
    return y


def bottleneck_downsample(x, output_depth, internal_scale=4, pad=(15, 20)):
    internal_depth = output_depth / internal_scale

    x_conv = Conv2D(internal_depth, (2, 2), stride=(2, 2))(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(internal_depth, (3, 3), padding="same")(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(output_depth, (1, 1), use_bias=False)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = SpatialDropout2D()(x_conv)

    x_pool = MaxPooling2D(pool_size=(2, 2))(x)
    x_pool = ZeroPadding2D(pad)(x_pool)

    x = Add()([x_conv, x_pool])
    y = Activation("relu")(x)
    return y


def bottleneck(x, output_depth, internal_scale=4, pad=(15, 20)):
    internal_depth = output_depth / internal_scale

    x_conv = Conv2D(internal_depth, (1, 1))(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(internal_depth, (3, 3), padding="same")(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(output_depth, (1, 1), use_bias=False)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = SpatialDropout2D()(x_conv)

    x_pool = MaxPooling2D(pool_size=(2, 2))(x)
    x_pool = ZeroPadding2D(pad)(x_pool)

    x = Add()([x_conv, x_pool])
    y = Activation("relu")(x)
    return y


def bottleneck_asymmetric(x, asymmetric, output_depth, internal_scale=4, pad=(15, 20)):
    internal_depth = output_depth / internal_scale

    x_conv = Conv2D(internal_depth, (1, 1))(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(internal_depth, (1, asymmetric), padding="same", use_bias=False)(x_conv)
    x_conv = Conv2D(internal_depth, (asymmetric, 1), padding="same")(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(output_depth, (1, 1), use_bias=False)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = SpatialDropout2D()(x_conv)

    x_pool = MaxPooling2D(pool_size=(2, 2))(x)
    x_pool = ZeroPadding2D(pad)(x_pool)

    x = Add()([x_conv, x_pool])
    y = Activation("relu")(x)
    return y


def bottleneck_dilated(x, dilated, output_depth, internal_scale=4, pad=(15, 20)):
    internal_depth = output_depth / internal_scale

    x_conv = Conv2D(internal_depth, (1, 1))(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(internal_depth, (3, 3), dilation_rate=(dilated, dilated), padding="same")(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(output_depth, (1, 1), use_bias=False)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = SpatialDropout2D()(x_conv)

    x_pool = MaxPooling2D(pool_size=(2, 2))(x)
    x_pool = ZeroPadding2D(pad)(x_pool)

    x = Add()([x_conv, x_pool])
    y = Activation("relu")(x)
    return y


def bottleneck_upsampling(x, dilated, output_depth, internal_scale=4, pad=(15, 20)):
    internal_depth = output_depth / internal_scale

    x_conv = Conv2D(internal_depth, (1, 1))(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2DTranspose(internal_depth, (3, 3), strides=(2, 2), padding="same")(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(output_depth, (1, 1), use_bias=False)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = SpatialDropout2D()(x_conv)

    x_pool = Conv2D(output_depth, (1, 1), use_bias=False)(x)
    x_pool = BatchNormalization()(x_pool)
    x_pool = UpSampling2D(size=(2, 2))(x_pool)

    x = Add()([x_conv, x_pool])
    y = Activation("relu")(x)
    return y


def TestNet(input_shape=(120, 160, 3), classes=21):
    inputs = Input(shape=input_shape)



    # 60 x 80 x 64

    # 30 x 40 x 128


    outputs = Activation("softmax")(x)
    model = Model(inputs, outputs)
    return model