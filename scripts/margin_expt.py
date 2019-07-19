#Imports
import ipdb
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

from adv_util import create_fully_connected_SVM_binary
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

"""
Setup
"""

#Configurations
#TODO: Implement experiments for different configurations
dataset = 'mnist'
num_classes = 2
top_model = 'SVM'
classes = [1,7]
measure_margin_method = 'geometric'
#Other options include confidence approach(like in Generalization lectures)
#Or the empirical approach - as described in Dawn's paper on Decision Boundary analysis"

#Get the dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train, y_test = y_train.astype('int'), y_test.astype('int')

#Truncate the dataset to only these two classes
#ones_train = np.argwhere(y_train == 1)[:, 0] 
#sevens_train = np.argwhere(y_train == 7)[:, 0] 
#indices_train = np.concatenate((ones_train, sevens_train))

indices_train = []

for idx, label in enumerate(y_train):
    if label == 1 or label == 7:
        indices_train.append(idx)

ones_test = np.argwhere(y_test== 1)[:, 0] 
sevens_test = np.argwhere(y_test == 7)[:, 0]
indices_test= np.concatenate((ones_test, sevens_test))

x_train, y_train, x_test, y_test = x_train[indices_train], y_train[indices_train], x_test[indices_test], y_test[indices_test]

print(x_train.shape)
print(x_test.shape)

#Replace with labels as 1's and -1's
for idx, label in enumerate(y_train):
    if y_train[idx] == 1:
        y_train[idx] = -1
    else:
        y_train[idx] = 1
 
for idx, label in enumerate(y_test):
    if y_test[idx] == 1:
        y_test[idx] = -1
    else:
        y_test[idx] = 1


x_train = x_train/255
x_test = x_test/255

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )

x_train_flat, input_shape = flatten_mnist(x_train)
x_test_flat, _ = flatten_mnist(x_test)


"""
Main Phase
Train models and record relevant info
"""

print('Setup done. Now training models')

#Create models with different depths
models_list = []
#num_hidden_range = range(1,10)
num_hidden_range = [1, 3, 5]

for num_hidden in num_hidden_range:
        model = create_fully_connected_SVM_binary(input_shape = input_shape, num_hidden = num_hidden, reg = 0.01)
        model.compile(optimizer="adadelta", loss=keras.losses.hinge, metrics=['accuracy'])    
        model.fit(x_train_flat, y_train, epochs = 30)
        models_list.append(model)

print('Trained the models. Attempting to analyze margins')

margins = []
accuracies = []
df = pd.DataFrame(y_train, columns = ['y'])

for (idx,model) in enumerate(models_list):
    #Margins
    last_layer = model.layers[-1]
    weights = last_layer.get_weights()
    norm = np.linalg.norm(weights[0])
    margin = 2./norm
    margins.append(margin)

    #Accuracies
    accuracies.append(model.evaluate(x_test_flat, y_test))

    #t-SNE representations 
    layer_output = model.layers[-2].output
    activation_model = models.Model(inputs=model.input, outputs=layer_output)
    featurized_x_train = activation_model.predict(x_train_flat)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(featurized_x_train)

    #Populate the dataframe
    columns = ['tsne-one-' + str(idx), 'tsne-two-' + str(idx)]
    df[columns[0]] = tsne_results[:, 0]
    df[columns[1]] = tsne_results[:, 1]

"""
Plotting and Printing
"""

ax1 = plt.subplot(2, 2, 1)
sns.scatterplot(
        x="tsne-one-0", y="tsne-two-0",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
)

ax2 = plt.subplot(2, 2, 2)
sns.scatterplot(
        x="tsne-one-1", y="tsne-two-1",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
)


ax3 = plt.subplot(2, 2, 3)
sns.scatterplot(
        x="tsne-one-2", y="tsne-two-2",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax3
)

plt.show()

print('Margins:')
print(margins)

print('Accuracies')
print(accuracies)
