# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Dropout
from keras import regularizers
from keras import backend
from keras.utils import to_categorical 

import sys
sys.path.insert(0, '../cleverhans')
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.loss import CrossEntropy

from adv_util import create_model_one, adv_evaluate, train_and_test
#Config
nb_epochs = 2
batch_size = 128
learning_rate = 0.001
#TODO: Figure out what these two quantities below are
train_dir = 'train_dir'
filename = 'mnist.ckpt'
filename_adv_trained = 'mnist_adv.ckpt'

#TODO: Right now the saving is not working as expected. Fix.
load_model = False
benign = False
adversarial = True
rng = np.random.RandomState([2018, 11, 25])
label_smoothing = 0.1

train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'train_dir': train_dir,
      'filename': filename
}

eval_params = {
        'batch_size': batch_size
}

fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}

#Setup checks that everything is ok
if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")
"""
if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
      "'th', temporarily setting to 'tf'")
"""

#Load the data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
example = train_images[0]
train_images = train_images/255.0
test_images = test_images/255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

n, img_rows, img_cols = train_images.shape
train_images = train_images.reshape(n, img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
num_classes = 10


"""
Create computation graph
"""

x = tf.placeholder(tf.float32, shape= (None, img_rows, img_cols, 1))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

if benign:
    #model = create_model_one(input_shape = input_shape, num_classes = num_classes)
    model = cnn_model(img_rows = img_rows, img_cols = img_cols, channels = 1, nb_filters = 64, nb_classes = num_classes)
    preds = model(x)
    print('Defined the model graph')

    """
    Phase 2.1: Train and evaluate on legit test examples
    """

    #Could do this the normal keras way. And not using the helper functions from cleverhans.
    wrap = KerasModelWrapper(model)
    loss = CrossEntropy(wrap, smoothing=label_smoothing)
    sess = tf.Session()
    keras.backend.set_session(sess)

    #Gets passed to train as a function handle
    def evaluate():
        acc = model_eval(sess, x, y, preds, test_images, test_labels, args = eval_params)
        print('Test accuracy of benign model on legit examples: %0.4f' % acc)

    saver = tf.train.Saver()
    path = os.path.join('..', train_dir, filename)
    if load_model:
        saver.restore(sess, path)
        print("Model loaded from: {}". format(path))
    else:
        train(sess, loss, train_images, train_labels, evaluate = evaluate, args = train_params, rng = rng)
        saver.save(sess, path)
        train_acc = model_eval(sess, x, y, preds, train_images, train_labels, args = eval_params)
        print('Train accuracy of benign model on legit examples: %0.4f' % train_acc)

    """
    Phase 2.2: Create the attack graph and evaluate on adversarial examples"
    """

    fgsm = FastGradientMethod(wrap, sess = sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model(adv_x)
    acc = model_eval(sess, x, y, preds_adv, test_images , test_labels, args=eval_params)
    print('Test accuracy of benign model on adversarial examples: %0.4f\n' %acc)

"""
Phase 2.3: Perform adversarial training and evaluate on both legit and adversarial
"""
#Define a new session for adversarial training
sess = tf.Session()
keras.backend.set_session(sess)

print("Repeating the process, using adversarial training")
# Redefine TF model graph
model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                  channels=1, nb_filters=64,
                  nb_classes=num_classes)
wrap_2 = KerasModelWrapper(model_2)
preds_2 = model_2(x)
fgsm2 = FastGradientMethod(wrap_2, sess=sess)

#Gets passed to CrossEntropy loss which leads to adversarial training
def attack(x):
    return fgsm2.generate(x, **fgsm_params)

preds_2_adv = model_2(attack(x))
loss_2 = CrossEntropy(wrap_2, smoothing=label_smoothing, attack=attack)

#Gets passed to train as a function handle
def evaluate_2():
    # Accuracy of adversarially trained model on legitimate test inputs
    accuracy = model_eval(sess, x, y, preds_2, test_images, test_labels,
                          args=eval_params)
    print('Test accuracy of adv trained on legitimate examples: %0.4f' % accuracy)

    # Accuracy of the adversarially trained model on adversarial examples
    accuracy = model_eval(sess, x, y, preds_2_adv, test_images,
                          test_labels, args=eval_params)
    print('Test accuracy of adv trained on adversarial examples: %0.4f' % accuracy)

# Perform and evaluate adversarial training
train(sess, loss_2, train_images, train_labels , evaluate=evaluate_2,
    args=train_params, rng=rng)

# Calculate training errors
accuracy = model_eval(sess, x, y, preds_2, train_images, train_labels,
                      args=eval_params)
accuracy = model_eval(sess, x, y, preds_2_adv, train_images,
                      train_labels, args=eval_params)

