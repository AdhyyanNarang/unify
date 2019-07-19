#Imports
import tensorflow as tf
import keras
from keras import models
import sys
import logging

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

print(tf.__version__)

from adv_util import create_fully_connected_SVM
#Dataset stuff
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Flattens the dataset
"""
def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )

x_train_flat, input_shape = flatten_mnist(x_train)
x_test_flat, _ = flatten_mnist(x_test)

#Create models with different depths
models = []
#num_hidden_range = range(2,11,2)
num_hidden_range = [4]

for num_hidden in num_hidden_range:
        model = create_fully_connected_SVM(input_shape = input_shape, num_classes = num_classes, num_hidden = num_hidden, reg = 0.01)
        model.compile(optimizer="adadelta", loss=keras.losses.categorical_hinge, metrics=['accuracy'])    
        model.fit(x_train_flat, y_train, epochs = 30)
        models.append(model)

print('Trained the models. Attempting to analyze margins')
#Analyze their margins
margins = []
for model in models:
    last_layer = model.layers[-1]
    weights = last_layer.get_weights()
    norm = np.linalg.norm(weights[0])
    margin = 1./norm
    margins.append(margin)

print('Margins:')
print(margins)

"""
I am not exactly sure how the norms printed actually relate to the margin.
Maybe another way would be to write a function to check the distance
to each of the hyperplanes - equivalent to looking at the scores in the 
penultimate layer
"""
#TODO: Figure out how to measure the geometric margins of the different models.
