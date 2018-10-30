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
# from mnist_cnn import Dummy,model

# load data
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

algorithm='neural_net'
num_features = 50 # number of features used in sparse linear model

np.random.seed(1)
img_size = 28
img_chan = 1
n_classes = 2


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





rho = 25
num_samples = 15000
kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
# may need to change the distance function to others ...
LIME = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=num_samples,
                                            return_mean=True, verbose=False, return_mapped=True)

predictions = y_pred_pris
predict_probas = y1
n_test = y_test.shape[0]
n_exp = 50
exp_ind = np.random.choice(range(n_test),n_exp,replace=False)#pris_wrong_index#range(n_exp)#

exps = {}
exps['pris'] = []
exps['adv'] = []
for i in range(n_exp):
    print('pristine data %s'%i)
    sys.stdout.flush()
    exp,mean,data_sample,label_sample = LIME.explain_instance(test_vectors[exp_ind[i],:].reshape((1,img_size*img_size)), 1, nn_predict_proba_fn, num_features)
    exps['pris'].append((exp,mean,data_sample,label_sample))
    print('avdersarial data %s'%i)
    exp, mean,data_sample,label_sample = LIME.explain_instance(adv_vectors[exp_ind[i],:].reshape((1,img_size*img_size)), 1, nn_predict_proba_fn, num_features)
    exps['adv'].append((exp,mean, data_sample,label_sample))


import pickle_workaround
# with open('interpret_37_samples50_nonbinary.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
pickle_workaround.pickle_dump([exps,exp_ind], "interpret_37_samples50_nonbinary_50feature.pkl")









# vec = tf.get_collection(tf.GraphKeys.VARIABLES)
# for i in range(len(vec)):
#   print(vec[i])
#
# graph = tf.get_default_graph()
# vec = sess.graph.get_operations()
# for i in range(len(vec)):
#   print(vec[i].name)
#
#

