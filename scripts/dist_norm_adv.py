#This is the same one as Experiment A in the Ipython notebook

#Various imports
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
from keras.callbacks import EarlyStopping

#Config
load_weights_flag = True 
save_weights_directory = "weights/"
num_hidden = 4
eps = 0.10
save_plot_directory = "img/tsne/robust_model"

if __name__ == '__main__':
    #Datset stuff
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    num_classes = 10
    y_train_old = y_train
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train_flat, input_shape = flatten_mnist(x_train)
    x_test_flat, _ = flatten_mnist(x_test)

    #Create and train models on regular and robust data
    sess = tf.keras.backend.get_session()
    keras.backend.set_session(sess)

    early_stop = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=4)

    model = create_fully_connected(input_shape = input_shape, num_classes = num_classes, num_hidden = num_hidden, reg = 0)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    if not load_weights_flag:
        model.fit(x_train_flat, y_train, epochs = 100, callbacks = [early_stop])
        model.save_weights(save_weights_directory + "weights_regular")
    else:
        model.load_weights(save_weights_directory + "weights_regular")

    x_test_flat_adv = adv_generate(sess, KerasModelWrapper(model), eps, x_test_flat)
    print(model.evaluate(x_test_flat, y_test))
    print(model.evaluate(x_test_flat_adv, y_test))

    x_train_flat_adv = adv_generate(sess, KerasModelWrapper(model), eps, x_train_flat)
    model_robust = create_fully_connected(input_shape = input_shape, num_classes = num_classes, num_hidden = num_hidden, reg = 0)
    model_robust.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    if not load_weights_flag:
        model_robust.fit(x_train_flat_adv, y_train, epochs = 100, callbacks = [early_stop])
        model_robust.save_weights(save_weights_directory + "weights_robust")
    else:
        model_robust.load_weights(save_weights_directory + "weights_robust")
    print(model_robust.evaluate(x_test_flat, y_test))
    print(model_robust.evaluate(x_test_flat_adv, y_test))

    #Dist split for regular model
    L = len(model.layers)
    layer_output = [model.layers[i].output for i in range(L-1)]
    activation_model = models.Model(inputs=model.input, outputs=layer_output)
    overall, correct, incorrect= dist_split(x_test_flat, x_test_flat_adv, y_test,  model, activation_model)

    #Dist split for adv model
    layer_output_robust = [model_robust.layers[i].output for i in range(L-1)]
    activation_model_robust = models.Model(inputs=model_robust.input, outputs=layer_output_robust)
    overall_rob, correct_rob, incorrect_rob = dist_split(x_test_flat, x_test_flat_adv, y_test, model_robust, activation_model_robust)

    #ipdb.set_trace()    

    tsne_activations = generate_data_to_plot(sess, model_robust, x_train_flat, y_train, eps = eps, tsne_flag = True)
    np.save("robust_ac", tsne_activations)
    for (idx, tsne_ac) in enumerate(tsne_activations):
            savepath = save_plot_directory + "/layer_" + str(idx)
            plot_data(tsne_ac, y_train_old, savepath)
