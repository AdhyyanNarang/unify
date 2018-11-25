# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Dropout
from keras import regularizers
from keras import backend

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper


#Load the data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

n, img_rows, img_cols = train_images.shape
train_images = train_images.reshape(n, img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
num_classes = 10

#Phase 1: Create computation graph
def create_model_one():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5),
                     activation = 'relu',
                     input_shape = input_shape,
                    ))
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3),
                     activation = 'relu',
                    ))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu',
                                 ))
    model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu',
                                 ))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation = 'relu',
                                ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1024, activation = 'relu',
                                ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation = 'softmax'))
    return model

def create_model_two():
    #TODO: Use the model that Ruoxi used here.
    return None


def adv_evaluate(model_input, epsilon_input):
    wrap = KerasModelWrapper(model_input)
    fgsm = FastGradientMethod(wrap)
    adv_x = fgsm.generate_np(test_images, eps = epsilon_input, clip_min = -2, clip_max = 2)
    preds_adv = model_input.predict(adv_x)
    eval_par = {'batch_size': 60000}
    test_loss, acc = model_input.evaluate(adv_x, test_labels)
    return acc, adv_x, preds_adv

#Uses adv examples from tensorflow_adversarial
def adv_evaluateV2(model_input, epsilon_input):
    return None

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

#Uses adv examples from tensorflow_adversarial
def train_and_testv2(model_input, X_input, y_input, epochs = 2, eps_test = 0.2):
    return None

model = create_model_one()
model.compile(optimizer = tf.train.AdamOptimizer(),
             loss = 'sparse_categorical_crossentropy',
             metrics =['accuracy'])

print('Compiled the model!')
print('Creating modified train set!')
#Phase 2.1: Create modified train set
with tf.Session() as sess:
    #Train the model
    sess.run(tf.global_variables_initializer())
    model.fit(train_images, train_labels, epochs = 2)
    """
    #Choose indices for normal and adversarial examples
    threshold = int(np.round(0.4*n))
    example_indices = range(threshold,n)
    train_indices = range(0, threshold)

    #Create the adversarial examples
    chosen_adv_images = train_images[example_indices]
    """
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess = sess)
    adv_x = fgsm.generate_np(train_images, eps = 0.3, clip_min = -1, clip_max = 1)

    #Create our new train set
    #new_train = np.vstack((train_images[train_indices], adv_x))
    new_train = adv_x

#Phase 2.2: Test both regaular and adversarially trained models
with tf.Session() as sess:
    #Train the model
    sess.run(tf.global_variables_initializer())
    print('Testing the benign model')
    train_and_test(model, train_images, train_labels)
    sess.run(tf.global_variables_initializer())
    layer = model.get_layer(index = 4)
    weights = layer.get_weights()
    print('Training the adversarially trained model')
    train_and_test(model, new_train, train_labels)
