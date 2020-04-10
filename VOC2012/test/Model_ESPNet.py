import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Activation, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU, Add, PReLU, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.losses import sparse_categorical_crossentropy

"""
https://github.com/sacmehta/ESPNet/blob/master/train/Model.py
H x W x D : 128 x 120 x 3 -> 128 x 120 x (class)
o_w = (in_w + 2 * pad - k_w - (k_w - 1)*(d_w - 1)) / s_w + 1
"""


def weighted_SparseCategoricalCrossentropy(sample_weights, classes=21):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.uint8)
        y_true = tf.one_hot(y_true, depth=classes)
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true * sample_weights
        # cce = tf.keras.losses.SparseCategoricalCrossentropy()
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(y_true, y_pred)
    return loss_function


def ESP_Module(x, out_depth, internal_depth_ratio=4, Momentum=0.01, Alpha=0.3, Add_flag=True):
    internal_depth = int(out_depth / internal_depth_ratio)
    internal_depth_e = out_depth - 3 * internal_depth

    x = Conv2D(internal_depth, (1, 1))(x)

    x_conv_d1 = ZeroPadding2D(padding=1)(x)
    x_conv_d1 = DepthwiseConv2D((3, 3), use_bias=False)(x_conv_d1)
    x_conv_d1 = Conv2D(internal_depth_e, (1, 1))(x_conv_d1)

    x_conv_d2 = DepthwiseConv2D((3, 3), dilation_rate=(2, 2), padding="same", use_bias=False)(x)
    x_conv_d2 = Conv2D(internal_depth, (1, 1))(x_conv_d2)

    x_conv_d4 = DepthwiseConv2D((3, 3), dilation_rate=(4, 4), padding="same", use_bias=False)(x)
    x_conv_d4 = Conv2D(internal_depth, (1, 1))(x_conv_d4)

    x_conv_d8 = DepthwiseConv2D((3, 3), dilation_rate=(8, 8), padding="same", use_bias=False)(x)
    x_conv_d8 = Conv2D(internal_depth, (1, 1))(x_conv_d8)

    add_1 = x_conv_d2
    add_2 = Add()([add_1, x_conv_d4])
    add_3 = Add()([add_2, x_conv_d8])

    x_conv = Concatenate(axis=3)([x_conv_d1, add_1, add_2, add_3])

    if Add_flag == True:
        x_conv = Add()([x_conv, x])

    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    output = LeakyReLU(alpha=Alpha)(x_conv)
    return output


def ESP_Module_DownSampling(x, out_depth, internal_depth_ratio=4, Momentum=0.01, Alpha=0.3, Add_flag=True):
    internal_depth = int(out_depth / internal_depth_ratio)
    internal_depth_e = out_depth - 3 * internal_depth

    x = Conv2D(internal_depth, (3, 3), stride=(2, 2))(x)

    x_conv_d1 = ZeroPadding2D(padding=1)(x)
    x_conv_d1 = DepthwiseConv2D((3, 3), use_bias=False)(x_conv_d1)
    x_conv_d1 = Conv2D(internal_depth_e, (1, 1))(x_conv_d1)

    x_conv_d2 = DepthwiseConv2D((3, 3), dilation_rate=(2, 2), padding="same", use_bias=False)(x)
    x_conv_d2 = Conv2D(internal_depth, (1, 1))(x_conv_d2)

    x_conv_d4 = DepthwiseConv2D((3, 3), dilation_rate=(4, 4), padding="same", use_bias=False)(x)
    x_conv_d4 = Conv2D(internal_depth, (1, 1))(x_conv_d4)

    x_conv_d8 = DepthwiseConv2D((3, 3), dilation_rate=(8, 8), padding="same", use_bias=False)(x)
    x_conv_d8 = Conv2D(internal_depth, (1, 1))(x_conv_d8)

    add_1 = x_conv_d2
    add_2 = Add()([add_1, x_conv_d4])
    add_3 = Add()([add_2, x_conv_d8])

    x_conv = Concatenate(axis=3)([x_conv_d1, add_1, add_2, add_3])

    x_conv = BatchNormalization(momentum=Momentum)(x_conv)
    output = LeakyReLU(alpha=Alpha)(x_conv)
    return output


def encoder(x, classes=5):

    pool_1 = MaxPooling2D(pool_size=(2, 2))(x)
    pool_2 = MaxPooling2D(pool_size=(4, 4))(x)

    # 128 x 128 x 3
    x = ZeroPadding2D(padding=1)(x)
    x = DepthwiseConv2D((3, 3), stride=(2, 2))(x)
    x = Conv2D(16, (1, 1))(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Concatenate(axsi=3)(x, pool_1)
    x_0 = x
    # 64 x 64 x 16

    x = ESP_Module_DownSampling(x, 64)
    x_skip = x
    for _ in range(2):
        x = ESP_Module(x, 64)
    x = Concatenate(axis=3)([x, x_skip])
    x = BatchNormalization(momentum=0.01)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Concatenate(axsi=3)(x, pool_2)
    x_1 = x
    # 32 x 32 x 128

    x = ESP_Module_DownSampling(x, 128)
    x_skip = x
    for _ in range(3):
        x = ESP_Module(x, 128)
    x = Concatenate(axis=3)([x, x_skip])
    x = BatchNormalization(momentum=0.01)(x)
    x = LeakyReLU(alpha=0.3)(x)
    # 16 x 16 x 256

    x = Conv2D(classes, (1, 1))(x)
    # 16 x 16 x classes
    return x, x_0, x_1


def TestNet(input_shape=(128, 128, 3), classes=21):

    inputs = Input(shape=input_shape)

    x, x_0, x_1 = encoder(inputs)
    x_0 = Conv2D(classes, (1, 1))(x_0)
    x_1 = Conv2D(classes, (1, 1))(x_1)
    # 16 * 16 * classes

    x = ZeroPadding2D(padding_size=(1, 1))(x)
    x = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization(momentum=0.01)(x)
    # 32 * 32 * classes

    x = Concatenate(axis=3)([x, x_1])
    x = BatchNormalization(momentum=0.01)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = ESP_Module(x, classes, Add_flag=False)
    x = ZeroPadding2D(padding_size=(1, 1))(x)
    x = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(x)
    # 64 * 64 * classes

    x = Concatenate(axis=3)([x, x_0])
    x = ZeroPadding2D(padding_size=(1, 1))(x)
    x = DepthwiseConv2D((3, 3))(x)
    x = Conv2D(classes, (1, 1))(x)
    x = BatchNormalization(momentum=0.01)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = ZeroPadding2D(padding_size=(1, 1))(x)
    x = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(x)
    # 128 * 128 * classes

    outputs = Activation("softmax")(x)
    model = Model(inputs, outputs)
    return model
