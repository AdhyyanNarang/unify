# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import regularizers

# Helper libraries
import numpy as np
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper


"""
Define util functions
"""

def create_model_one(input_shape, num_classes, logits = False, input_ph = None):
    model = keras.Sequential()
    layers = [keras.layers.Conv2D(32, (5,5), input_shape = input_shape),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(32, kernel_size = (3,3)),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size = (2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3,3)),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(64, (3,3)),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size = (2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(1024),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1024),
            keras.layers.Activation('relu'),
            keras.layers.Dense(num_classes)]
    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    model.add(keras.layers.Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

def adv_evaluate(model_input, epsilon_input):
    wrap = KerasModelWrapper(model_input)
    fgsm = FastGradientMethod(wrap)
    adv_x = fgsm.generate_np(test_images, eps = epsilon_input, clip_min = -2, clip_max = 2)
    preds_adv = model_input.predict(adv_x)
    eval_par = {'batch_size': 60000}
    test_loss, acc = model_input.evaluate(adv_x, test_labels)
    return acc, adv_x, preds_adv

#Uses adversarial examples from cleverhans
def train_and_test(model_input, X_input, y_input, epochs = 2, eps_test = 0.2):
    model_input.fit(X_input, y_input, epochs = epochs)
    print('finished training')
    test_loss, test_acc = model_input.evaluate(test_images, test_labels)
    #Evaluation of normal model on adversarial test examples
    print('Test accuracy:', test_acc)
    #Evaluation of adversarially trained model
    acc, adv_x_np, preds = adv_evaluate(model_input, 0.2)
    print('Test accuracy on adversarial examples:', acc)


