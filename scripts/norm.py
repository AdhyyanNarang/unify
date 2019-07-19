import tensorflow as tf
import keras
from keras import models
import sys
import logging
sys.path.insert(0, '../')

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from mnist_corruption import gaussian_blurring, corrupt_data, random_perturbation, random_blackout_whiteout
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from adv_util import * 
import ipdb
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



x_train_flat, input_shape = flatten_mnist(x_train)
x_test_flat, _ = flatten_mnist(x_test)

sess = tf.keras.backend.get_session()
keras.backend.set_session(sess)


#num_hidden_range = range(2, 11, 2)
num_hidden_range = [2,10]
print(num_hidden_range)
regular_metrics = []
#blur_metrics = []
adv_metrics = []
models_saved = []
norms_list = []
early_stop = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=4)


for num_hidden in num_hidden_range:
    print(num_hidden)
    model = create_fully_connected(input_shape = input_shape, num_classes = num_classes, num_hidden = num_hidden, reg = 0)
    print(model.summary())
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    sess.run(tf.global_variables_initializer())
    model.fit(x_train_flat, y_train, epochs = 100, callbacks = [early_stop])
    models_saved.append(model)
    regular_metrics.append(model.evaluate(x_test_flat, y_test))
    #blur_metrics.append(model.evaluate(x_test_blur_flat, y_test))
    x_test_adv_flat = adv_generate(sess, KerasModelWrapper(model), 0.05, x_test_flat)
    adv_metrics.append(model.evaluate(x_test_adv_flat, y_test))

    model_norms = []
    for layer in model.layers:
        weights = layer.get_weights()
        norm = sum_entries_norm(weights[0])
        model_norms.append(norm)

    norms_list.append(model_norms)

ipdb.set_trace()
