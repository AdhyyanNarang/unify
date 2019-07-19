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

from mnist_corruption import gaussian_perturbation, random_perturbation
print(tf.__version__)
import ipdb

from adv_util import create_fully_connected

#Configurations
model_weights_path = 'model_weights.h5'
load_weights_flag = True 
num_adv = 100
eps = 0.1
num_iter = 10000

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

#Create model and train it
model = create_fully_connected(input_shape = input_shape, num_classes = num_classes, num_hidden = 4, reg = 0)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
if load_weights_flag:
    model.load_weights(model_weights_path)
else:
    model.fit(x_train_flat, y_train, epochs = 15)
    model.save_weights(model_weights_path)

#Try to create adversarial test images
def create_adv_sample(model, eps, x_test, y_test):
    adv_images = []
    success_count = 0
    for idx, image in enumerate(x_test):
        #ipdb.set_trace()
        image = image.reshape(1, 784)
        
        print(idx)
        #If it's already incorrectly classified, don't bother
        y_pred = model.predict(image)
        if np.argmax(y_pred) != np.argmax(y_test[idx]):
            adv_images.append(image)
            continue

        #Sample in 100 random directions to find an error
        counter = 0
        perturbed_image = image
        while counter < num_iter:
            #perturbed_image = gaussian_perturbation(image, eps)
            perturbed_image = random_perturbation(image, eps)
            y_pred = model.predict(perturbed_image)
            if np.argmax(y_pred) != np.argmax(y_test[idx]):
                success_count += 1
                break
            counter += 1
        adv_images.append(perturbed_image)
    return adv_images, success_count

gaussian_test = gaussian_perturbation(x_test_flat)
acc = model.evaluate(gaussian_test, y_test)
print(acc)

"""
test_image = gaussian_test[0]
two_d = test_image.reshape((28,28))
im = plt.imshow(two_d, cmap = plt.cm.binary)
plt.show()
"""


adv_images, success_count = create_adv_sample(model, eps, x_test_flat[:num_adv], y_test[:num_adv])


print('Success count')
print(success_count)

test_image = adv_images[0]
two_d = test_image.reshape((28,28))
plt.imshow(two_d, cmap = plt.cm.binary)
#plt.show()

