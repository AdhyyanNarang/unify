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

rho = 25
num_samples = 15000
n_test = X_test.shape[0]
n_exp = 100

# defense strategies
with open('interpret_37_neighbors_stat.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    conf_mean_pris, conf_std_pris, conf_mean_adv, conf_std_adv, exp_ind = pickle.load(f)



plt.hist(np.abs(conf_mean_pris-0.5),label='pristine neighbors')
plt.hist(np.abs(conf_mean_adv-0.5),label='adversarial neighbors')
plt.legend()
plt.show()

plt.hist(conf_std_pris,label='pristine')
plt.hist(conf_std_adv,label='adversarial')
plt.legend()
plt.show()

plt.hist(np.abs(conf_mean_pris-0.5)/conf_std_pris,label='pristine neighbors')
plt.hist(np.abs(conf_mean_adv-0.5)/conf_std_adv,label='adversarial neighbors')
plt.legend()
plt.show()

# defense strategies
with open('interpret_37_neighbors_count.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    count_pris, count_adv, exp_ind = pickle.load(f)

ingredient_pris = np.empty(len(exp_ind))
ingredient_adv= np.empty(len(exp_ind))
for i in range(len(exp_ind)):
    ingredient_pris[i] = count_pris[i,y_pred_pris[exp_ind[i]]]/num_samples
    ingredient_adv[i] = count_adv[i, y_pred_adv[exp_ind[i]]]/num_samples

# ingredient_pris = count_pris[:,0]/num_samples
# ingredient_adv = count_adv[:,0]/num_samples

plt.hist(ingredient_pris,alpha=0.5,label='pristine',normed=True)
plt.hist(ingredient_adv,alpha=0.5,label='adversarial',normed=True)
plt.xlabel('Proportion of X\'s neighboring points with the same label as X')
plt.ylabel('Normalized histogram')
plt.legend()
plt.show()

# majority vote defense
pred_gt = y_test_compact[exp_ind]
pred_mv = np.empty(len(exp_ind))
pred_mv = np.argmax(count_adv,axis=1)
acc_mv = np.sum(pred_gt == pred_mv)/len(exp_ind)
acc_adv = np.sum(pred_gt == y_pred_adv[exp_ind])/len(exp_ind)
print(acc_mv)
print(acc_adv)


# precision and recall
threshold = np.arange(0.3,1,0.1)
ingredient = np.concatenate((ingredient_adv,ingredient_pris))
decision_gt = np.concatenate((np.ones(n_exp),np.zeros(n_exp)))
precision = []
recall = []
for i in range(len(threshold)):
    th = threshold[i]
    decision = (ingredient < th)
    tp = np.sum(np.logical_and(decision==1,decision_gt==1))
    fp = np.sum(np.logical_and(decision==1,decision_gt==0))
    fn = np.sum(np.logical_and(decision==0,decision_gt==1))
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))

plt.plot(threshold,precision,label='precision')
plt.plot(threshold,recall,label='recall')
plt.xlabel('threshold below which the example is considered adversarial')
plt.legend()


# covariance defense
with open('interpret_37_neighbors_corr.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    corr_pris, corr_adv, exp_ind = pickle.load(f)

corr_mean_pris = np.empty(len(exp_ind))
corr_mean_adv= np.empty(len(exp_ind))
for i in range(len(exp_ind)):
    corr_mean_pris[i] = np.mean(np.diag(corr_pris[i]))
    corr_mean_adv[i] = np.mean(np.diag(corr_adv[i]))

# corr_mean_pris = np.empty(len(exp_ind))
# corr_mean_adv= np.empty(len(exp_ind))
# for i in range(len(exp_ind)):
#     corr_mean_pris[i] = np.sum(corr_pris[i])- np.sum(np.diag(corr_pris[i]))
#     corr_mean_adv[i] = np.sum(corr_adv[i])- np.sum(np.diag(corr_adv[i]))

# ingredient_pris = count_pris[:,0]/num_samples
# ingredient_adv = count_adv[:,0]/num_samples

plt.hist(corr_mean_pris,label='pristine',normed=True)
plt.hist(corr_mean_adv,label='adversarial',normed=True)
plt.xlabel('sum of entries in the covariance matrix')
plt.ylabel('normalized histogram')
plt.legend()
plt.show()


idx = 1
plt.figure()
plt.imshow(corr_pris[idx])
plt.xlabel('benign examples')
plt.figure()
plt.imshow(corr_adv[idx])
plt.xlabel('adversarial examples')

# conf_mean_pris = np.empty(n_exp)
# conf_mean_adv = np.empty(n_exp)
# conf_std_pris = np.empty(n_exp)
# conf_std_adv = np.empty(n_exp)
#
# for i in range(n_exp):
#     conf_mean_pris[i] = np.mean(X_pris_label[i][:,1])
#     conf_std_pris[i] = np.std(X_pris_label[i][:, 1])
#     conf_mean_adv[i] = np.mean(X_adv_label[i][:, 1])
#     conf_std_adv[i] = np.std(X_adv_label[i][:, 1])
#


## number of nonzeros
count_nonzero_test = np.empty(n_test)
count_nonzero_adv = np.empty(n_test)
for i in range(n_test):
    count_nonzero_test[i] = np.count_nonzero(test_vectors[i,:])
    count_nonzero_adv[i] = np.count_nonzero(adv_vectors[i,:])
    print(i)

plt.hist(count_nonzero_test,label='test')
# plt.hist(count_nonzero_adv,label='adv')
plt.xlabel('number of nonzero entries')
plt.ylabel('histogram')
plt.title('attack - fast gradient sign method')
plt.legend()
plt.show()




z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

X_tmp = np.empty((n_classes, 28, 28))
y_tmp = np.empty((n_classes, n_classes))
for i in range(n_classes):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
    cur = np.random.choice(ind)
    X_tmp[i] = np.squeeze(X_adv[cur])
    y_tmp[i] = y2[cur]

print('\nPlotting results')

fig = plt.figure(figsize=(n_classes, 1.2))
gs = gridspec.GridSpec(1, n_classes, wspace=0.05, hspace=0.05)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)
for i in range(n_classes):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                  fontsize=12)

print('\nSaving figure')


gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/deepfool_mnist_37.png')