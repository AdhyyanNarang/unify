import sys
import copy
import os
import numpy as np
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
from explainers import data_labels_distances_mapping_text


# load image data
with open('objs_37.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    X_train, y_train, X_test, y_test, y1, X_adv, y2 = pickle.load(f)

train_vectors = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
test_vectors = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
adv_vectors = X_adv.reshape((X_adv.shape[0],X_adv.shape[1]*X_adv.shape[2]))

y_pred_pris = np.argmax(y1,axis=1)
y_test_compact = np.argmax(y_test,axis=1)
acc_pris = sum(y_test_compact - y_pred_pris == 0)/len(y_test_compact)

y_pred_adv = np.argmax(y2,axis=1)
acc_adv = sum(y_test_compact - y_pred_adv == 0)/len(y_test_compact)
adv_wrong_index = np.where(y_test_compact!= y_pred_adv)[0]
adv_correct_index = np.where(y_test_compact==y_pred_adv)[0]
pris_correct_index = np.where(y_test_compact == y_pred_pris)[0]
pris_wrong_index = np.where(y_test_compact != y_pred_pris)[0]

np.random.seed(1)
img_size = 28
img_chan = 1
n_classes = 2



# load trained neural network
sess = tf.Session()
saver = tf.train.import_meta_graph('./model_37/mnist.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model_37'))
graph = tf.get_default_graph()
class Dummy:
    pass

env = Dummy()
env.x = graph.get_tensor_by_name("model/x:0")
env.ybar = graph.get_tensor_by_name("model/ybar:0")


def nn_predict_proba_fn(X_data):
    # n_classes = env.ybar.get_shape().as_list()[1]
    # n_sample = X_data.shape[0]
    yval = sess.run(env.ybar, feed_dict={env.x: X_data})
    return yval

def compute_correlation(X_data):
    n = X_data.shape[0]
    X_vectors = X_data.reshape((X_data.shape[0],X_data.shape[1]*X_data.shape[2]))
    corr = np.cov(np.transpose(X_vectors))
    return corr


rho = 25
num_samples = 15000

kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
# may need to change the distance function to others ...

n_test = X_test.shape[0]
n_exp = 100
exp_ind = np.random.choice(np.arange(n_test),n_exp,replace=False)
X_pris_neighbor = []
X_pris_label = []
X_adv_neighbor = []
X_adv_label = []
conf_mean_pris = np.empty(n_exp)
conf_mean_adv = np.empty(n_exp)
conf_std_pris = np.empty(n_exp)
conf_std_adv = np.empty(n_exp)

count_pris = np.empty((n_exp,2))
count_adv = np.empty((n_exp,2))
corr_pris = []
corr_adv = []

# generate neighborhood data
for i in range(n_exp):

    _, labels, _, _, data = data_labels_distances_mapping_text(test_vectors[exp_ind[i],:].reshape((1,img_size*img_size)),nn_predict_proba_fn,num_samples)
    corr = compute_correlation(data)
    corr_pris.append(corr)
    # count_pris[i,0] = np.sum(labels[:,0]> 0.5)
    # count_pris[i,1] = np.sum(labels[:,1]>=0.5)
    # conf_mean_pris[i] = np.mean(labels[:,1])
    # conf_std_pris[i] = np.std(labels[:,1])
    # X_pris_neighbor.append(data)
    # X_pris_label.append(labels)

    _, labels, _, _, data = data_labels_distances_mapping_text(adv_vectors[exp_ind[i], :].reshape((1,img_size*img_size)), nn_predict_proba_fn, num_samples)
    corr = compute_correlation(data)
    corr_adv.append(corr)
    # count_adv[i,0] = np.sum(labels[:,0]> 0.5)
    # count_adv[i,1] = np.sum(labels[:,1]>=0.5)
    # X_adv_neighbor.append(data)
    # X_adv_label.append(labels)
    # conf_mean_adv[i] = np.mean(labels[:, 1])
    # conf_std_adv[i] = np.std(labels[:,1])
    print(i)


with open('interpret_37_neighbors_corr.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([corr_pris,corr_adv,exp_ind], f)


