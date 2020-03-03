from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img, array_to_img
import tensorflow as tf
import numpy as np
import json

# input image dimensions
img_h, img_w = 5, 5

# the data, split between train and test sets
input_shape_keras = (img_h, img_w, 1)

model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 3),
                          activation='relu',
                          padding='same',
                          input_shape=input_shape_keras))

model.build()
model.summary()

weights = model.get_weights()
for w in weights:
    print(w.shape, w.dtype)
conv_w_0 = np.zeros(weights[0].shape)
conv_b = np.zeros(weights[1].shape)
input_img = np.zeros(input_shape_keras)

conv_w_0 = conv_w_0.transpose(3, 2, 0, 1)  # from(height, width, in_depth, out_depth) to (out_depth, in_depth, height, width)
conv_w_0[0][0][2][2] = 1
print(conv_w_0)
conv_w_0 = conv_w_0.transpose(2, 3, 1, 0)


for h in range(img_h):
    for w in range(img_w):
        for d in range(1):
            input_img[h][w][d] = (h + 1) * (w + 1)

input_img_keras = input_img
input_img_keras = input_img_keras.reshape((1,) + input_shape_keras)
print("weights[0].shape : ", weights[0].shape)
print("weights[1].shape : ", weights[1].shape)
print("input_img.shape : ", input_img.shape)
print("input_img_keras.shape : ", input_img_keras.shape)

test_weights = []
test_weights.append(conv_w_0)
test_weights.append(conv_b)
model.set_weights(test_weights)

output_img_keras = model.predict(input_img_keras)
output_img = output_img_keras.transpose(0, 3, 1, 2)

print(input_img.transpose(2, 0, 1))
print(output_img)

print("output_img_keras.shape", output_img_keras.shape)
print("output_img.shape : ", output_img.shape)

model.save('keras_conv2d_test.h5')







MODEL_TFLITE = "keras_conv2d_test.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS]
tfmodel = converter.convert() 
with open (MODEL_TFLITE, "wb") as m:
    m.write(tfmodel)

interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = input_img.reshape((1,) + input_img.shape).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.transpose(0, 3, 1, 2))