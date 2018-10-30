import numpy as np
import os
import scipy as sp
import json
import random
import sklearn
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import pickle
import explainers
import argparse
import collections
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

img_size = 28
img_chan = 1
n_classes = 2


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train_all, y_train_all), (X_test_all, y_test_all) = mnist.load_data()
select_index = [3,7]

ind_train = np.where(np.logical_or(y_train_all==select_index[0],y_train_all==select_index[1]))[0]
ind_test = np.where(np.logical_or(y_test_all==select_index[0],y_test_all==select_index[1]))[0]
X_train=X_train_all[ind_train,:,:]
y_train=y_train_all[ind_train]
y_train[y_train==select_index[0]] = 0
y_train[y_train==select_index[1]] = 1
X_test=X_test_all[ind_test,:,:]
y_test=y_test_all[ind_test]
y_test[y_test==select_index[0]] = 0
y_test[y_test==select_index[1]] = 1


X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train,num_classes=n_classes)
y_test = to_categorical(y_test,num_classes=n_classes)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]


sess = tf.Session()
saver = tf.train.import_meta_graph('./model_37_mixup/mnist.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model_37_mixup'))
graph = tf.get_default_graph()
class Dummy:
    pass

env = Dummy()
env.x = graph.get_tensor_by_name("model/x:0")
env.ybar = graph.get_tensor_by_name("model/ybar:0")


def nn_predict_proba_fn(X_data):
    yval = sess.run(env.ybar, feed_dict={env.x: X_data})
    return yval

## training examples
y_train_compact = np.argmax(y_train,axis=1)
ind_all_3 = np.where(y_train_compact == 0)[0]
ind_all_7 = np.where(y_train_compact == 1)[0]


# ind = 1
# ind3 = ind_all_3[ind]
# ind7 = ind_all_7[ind]
n_sample = 500
ind3_sub = np.random.choice(ind_all_3,n_sample)
ind7_sub = np.random.choice(ind_all_7,n_sample)
res = 0.1
ss = np.arange(0,1+res,res)
prob_sub = np.empty((n_sample,len(ss)))

for j in range(n_sample):
    ind3 = ind3_sub[j]
    ind7 = ind7_sub[j]
    X_sample_3 = X_train[ind3,:,:,:]
    X_sample_7 = X_train[ind7,:,:,:]

    # gs = gridspec.GridSpec(1,2, wspace=0.05, hspace=0.05)
    # fig = plt.figure(figsize=(2,1.2))
    # ax = fig.add_subplot(gs[0, 0])
    # ax.imshow(X_sample_3[:,:,0], cmap='gray', interpolation='none')
    #
    # ax = fig.add_subplot(gs[0,1])
    # ax.imshow(X_sample_7[:,:,0], cmap='gray', interpolation='none')


    imgs = []
    probs = []
    # gs = gridspec.GridSpec(1,len(ss), wspace=0.05, hspace=0.05)
    # fig = plt.figure(figsize=(2,1.2))
    for i in range(len(ss)):
        s = ss[i]
        img = X_sample_3 * (1-s) + X_sample_7 * s
        img = img.reshape((1,28,28,1))
        prob = nn_predict_proba_fn(img)[0][0]
        prob_sub[j,i] = prob
        # ax = fig.add_subplot(gs[0, i])
        # ax.imshow(img[0,:, :, 0], cmap='gray', interpolation='none')
        # imgs.append(img)
        # probs.append(prob)
    print(j)


plt.plot(ss,np.transpose(prob_sub),alpha=0.5)

prob_linear = np.empty((n_sample,len(ss)))
for i in range(len(ss)):
    s = ss[i]
    prob_linear[:,i] = prob_sub[:,0]* (1-s) + prob_sub[:,-1] * s

plot_num = 5
for i in range(plot_num):
    plt.plot(prob_linear[i,:],prob_sub[i,:],alpha= 0.5)


# test examples
y_test_compact = np.argmax(y_test,axis=1)
ind_all_3 = np.where(y_test_compact == 0)[0]
ind_all_7 = np.where(y_test_compact == 1)[0]



n_sample = 500
ind3_sub = np.random.choice(ind_all_3,n_sample)
ind7_sub = np.random.choice(ind_all_7,n_sample)
res = 0.01
ss = np.arange(0,1+res,res)
prob_sub = np.empty((n_sample,len(ss)))

for j in range(n_sample):
    ind3 = ind3_sub[j]
    ind7 = ind7_sub[j]
    X_sample_3 = X_test[ind3,:,:,:]
    X_sample_7 = X_test[ind7,:,:,:]

    # gs = gridspec.GridSpec(1,2, wspace=0.05, hspace=0.05)
    # fig = plt.figure(figsize=(2,1.2))
    # ax = fig.add_subplot(gs[0, 0])
    # ax.imshow(X_sample_3[:,:,0], cmap='gray', interpolation='none')
    #
    # ax = fig.add_subplot(gs[0,1])
    # ax.imshow(X_sample_7[:,:,0], cmap='gray', interpolation='none')


    imgs = []
    probs = []
    # gs = gridspec.GridSpec(1,len(ss), wspace=0.05, hspace=0.05)
    # fig = plt.figure(figsize=(2,1.2))
    for i in range(len(ss)):
        s = ss[i]
        img = X_sample_3 * (1-s) + X_sample_7 * s
        img = img.reshape((1,28,28,1))
        prob = nn_predict_proba_fn(img)[0][0]
        prob_sub[j,i] = prob
        # ax = fig.add_subplot(gs[0, i])
        # ax.imshow(img[0,:, :, 0], cmap='gray', interpolation='none')
        # imgs.append(img)
        # probs.append(prob)
    print(j)


plt.plot(ss,np.transpose(prob_sub),alpha=0.5)



prob_linear = np.empty((n_sample,len(ss)))
for i in range(len(ss)):
    s = ss[i]
    prob_linear[:,i] = prob_sub[:,0]* (1-s) + prob_sub[:,-1] * s

plot_num = 100
plt.figure()
for i in range(plot_num):
    plt.plot(prob_linear[i,:],prob_sub[i,:],alpha= 0.5)
plt.xlabel('ground truth probability')
plt.ylabel('neural network output')
plt.title('after data augmentation')
plt.show()

## gradual change of samples

ind3_sub = np.random.choice(ind_all_3,n_sample)
ind7_sub = np.random.choice(ind_all_7,n_sample)
res = 0.1
ss = np.arange(0,1+res,res)
prob_sub = np.empty((n_sample,len(ss)))

j = 0
ind3 = ind3_sub[j]
ind7 = ind7_sub[j]
X_sample_3 = X_test[ind3,:,:,:]
X_sample_7 = X_test[ind7,:,:,:]

gs = gridspec.GridSpec(1,2, wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(2,1.2))
ax = fig.add_subplot(gs[0, 0])
ax.imshow(X_sample_3[:,:,0], cmap='gray', interpolation='none')

ax = fig.add_subplot(gs[0,1])
ax.imshow(X_sample_7[:,:,0], cmap='gray', interpolation='none')


imgs = []
probs = []
gs = gridspec.GridSpec(1,len(ss), wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(2,1.2))
for i in range(len(ss)):
    s = ss[i]
    img = X_sample_3 * (1-s) + X_sample_7 * s
    img = img.reshape((1,28,28,1))
    prob = nn_predict_proba_fn(img)[0][0]
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img[0,:, :, 0], cmap='gray', interpolation='none')
print(j)