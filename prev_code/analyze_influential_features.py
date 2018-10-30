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

algorithm='neural_net'
num_features = 100 # number of features used in sparse linear model

np.random.seed(1)
img_size = 28
img_chan = 1
n_classes = 2




# load interpretability experiment results

with open('interpret_37_samples50_nonbinary.pkl', 'rb') as f:
    exps,exp_ind= pickle.load(f)

predictions = y_pred_pris
predict_probas = y1
n_test = y_test.shape[0]
n_exp = 50
select_index = [3,7]


#####################
# 0: black, 1: white
#####################

## prediction accuracy on the selected sample

acc_se_pris = sum(y_test_compact[exp_ind] - y_pred_pris[exp_ind]==0)/n_exp
acc_se_adv = sum(y_test_compact[exp_ind] - y_pred_adv[exp_ind]==0)/n_exp
print('benign acc is %s' % acc_se_pris)
print('adversarial acc is %s' % acc_se_adv)



## plot important feature for adversarial image and pristine images
n_plot = 10
plot_ind = exp_ind[0:n_plot]

gs = gridspec.GridSpec(2, n_plot, wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(2, 1.2))

# plot test image and the adversarially perturbed one

for i in range(n_plot):
    plot_i = plot_ind[i]
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_test[plot_i,:,:,0], cmap='gray', interpolation='none')
    ax.set_xlabel('{0} ({1:.2f})'.format(select_index[y_pred_pris[plot_i]], y1[plot_i,y_pred_pris[plot_i]]),
                  fontsize=12)
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(X_adv[plot_i,:,:,0], cmap='gray', interpolation='none')
    ax.set_xlabel('{0} ({1:.2f})'.format(select_index[y_pred_adv[plot_i]], y2[plot_i,y_pred_adv[plot_i]]),
                  fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
plt.show()


# plot influential features for benign and adversarial examples
fig = plt.figure(figsize=(2, 1.2))
x_test_feat = np.empty((n_exp,img_size,img_size))
x_adv_feat = np.empty((n_exp,img_size,img_size))
for i in range(n_plot):
    # plot_i = plot_ind[i]
    plot_i = np.where(exp_ind == plot_ind[i])[0][0]
    import_region_pris = np.zeros(img_size * img_size)
    for j in range(len(exps['pris'][plot_i][0])):
        import_region_pris[exps['pris'][plot_i][0][j][0]] = 1
    import_region_pris = import_region_pris.reshape((img_size,img_size))
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(import_region_pris, cmap='gray', interpolation='none')


    import_region_adv = np.zeros(img_size*img_size)
    for j in range(len(exps['adv'][plot_i][0])):
        import_region_adv[exps['adv'][plot_i][0][j][0]] = 1
    import_region_adv = import_region_adv.reshape((img_size,img_size))
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(import_region_adv, cmap='gray', interpolation='none')

    x_test_feat[i,:,:] = import_region_pris
    x_adv_feat[i,:,:] = import_region_adv

ax.set_xticks([])
ax.set_yticks([])
plt.show()

## compute differential of an image
def compute_difference(img):
    w,h = img.shape
    diff = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            if i != 0:
                diff[i, j] += img[i, j] - img[i - 1, j]
            if i != w - 1:
                diff[i, j] += img[i, j] - img[i + 1, j]
            if j!=0:
                diff[i,j] += img[i,j] - img[i,j-1]
            if j != h-1:
                diff[i,j] += img[i,j] - img[i,j+1]
    return diff

x_test_feat = np.empty((n_exp,img_size,img_size))
x_adv_feat = np.empty((n_exp,img_size,img_size))
for i in range(n_exp):
    plot_i = i
    import_region_pris = np.zeros(img_size * img_size)
    for j in range(len(exps['pris'][plot_i][0])):
        import_region_pris[exps['pris'][plot_i][0][j][0]] = 1
    import_region_pris = import_region_pris.reshape((img_size,img_size))

    import_region_adv = np.zeros(img_size*img_size)
    for j in range(len(exps['adv'][plot_i][0])):
        import_region_adv[exps['adv'][plot_i][0][j][0]] = 1
    import_region_adv = import_region_adv.reshape((img_size,img_size))

    x_test_feat[i,:,:] = import_region_pris
    x_adv_feat[i,:,:] = import_region_adv

x_test_diff = np.empty((x_test_feat.shape))
x_adv_diff = np.empty((x_adv_feat.shape))
for i in range(n_exp):
    x_test_diff[i,:,:] = compute_difference(x_test_feat[i,:,:])
    x_adv_diff[i,:,:] = compute_difference(x_adv_feat[i,:,:])

fig = plt.figure(figsize=(2, 1.2))
for i in range(n_plot):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(x_test_diff[i,:,:], cmap='gray', interpolation='none')

    ax = fig.add_subplot(gs[1, i])
    ax.imshow(x_adv_diff[i,:,:], cmap='gray', interpolation='none')

ax.set_xticks([])
ax.set_yticks([])
plt.show()

energy_test = np.empty(n_exp)
energy_adv = np.empty(n_exp)
for i in range(n_exp):
    energy_test[i] = np.sum(np.abs(x_test_diff[i,:,:]))
    energy_adv[i] = np.sum(np.abs(x_adv_diff[i,:,:]))

plt.hist(energy_test,label='benign')
plt.hist(energy_adv,label='adversarial')
plt.xlabel('differential image energy')
plt.legend()
plt.show()
# plot coefficients in front of influential features for benign and adverarial examples
for plot_i in range(n_exp):
    plt.figure()
    num_features = len(exps['pris'][plot_i ][0])
    coef_pris = np.empty(num_features)
    coef_adv = np.empty(num_features)
    for i in range(num_features):
        coef_pris[i] = exps['pris'][plot_i ][0][i][1]
        coef_adv[i] = exps['adv'][plot_i ][0][i][1]

    plt.hist(coef_pris,alpha=0.5,label='benign')
    plt.hist(coef_adv,alpha=0.5,label='adv')
    plt.legend()
    plt.show()

coef_var_pris = np.empty(n_exp)
coef_var_adv = np.empty(n_exp)

for i in range(n_exp):
    temp_pris = np.empty(num_features-2)
    temp_adv = np.empty(num_features-2)
    for j in range(num_features-2):
        temp_pris[j] = exps['pris'][i][0][j][1]
        temp_adv[j] = exps['adv'][i][0][j][1]
    coef_var_pris[i] = np.std(temp_pris)
    coef_var_adv[i] = np.std(temp_adv)
plt.hist(coef_var_pris,label='benign')
plt.hist(coef_var_adv,label='adversarial')
plt.legend()




## plot neighborhood of an instance
n_plot_neighbor = 10
num_samples = 15000
gs = gridspec.GridSpec(2, n_plot_neighbor+1, wspace=0.05, hspace=0.05)
fig = plt.figure(figsize=(2,1.2))
rand_ind = np.random.choice(np.arange(num_samples),n_plot_neighbor,replace=False)
pred_i = 0
for i in range(n_plot_neighbor):
    ax=fig.add_subplot(gs[0,i])
    ax.imshow(exps['pris'][pred_i][2][rand_ind[i],:,:,0], cmap='gray', interpolation='none')
    ax.set_xlabel('{0} ({1:.2f})'.format(select_index[np.argmax(exps['pris'][pred_i][3][rand_ind[i]])],
                                         np.max(exps['pris'][pred_i][3][rand_ind[i]])),
                  fontsize=12)
    ax=fig.add_subplot(gs[1, i])
    ax.imshow(exps['adv'][pred_i][2][rand_ind[i],:,:,0], cmap='gray', interpolation='none')
    ax.set_xlabel('{0} ({1:.2f})'.format(select_index[np.argmax(exps['adv'][pred_i][3][rand_ind[i]])],
                                         np.max(exps['adv'][pred_i][3][rand_ind[i]])),
                  fontsize=12)

ax = fig.add_subplot(gs[0, n_plot_neighbor])
ax.imshow(X_test[exp_ind[pred_i], :, :, 0], cmap='gray', interpolation='none')
ax.set_xlabel('{0} ({1:.2f})'.format(select_index[y_pred_pris[exp_ind[pred_i]]], y1[exp_ind[pred_i], y_pred_pris[exp_ind[pred_i]]]),
              fontsize=12)
ax = fig.add_subplot(gs[1, n_plot_neighbor])
ax.imshow(X_adv[exp_ind[pred_i],:,:,0], cmap='gray', interpolation='none')
ax.set_xlabel('{0} ({1:.2f})'.format(select_index[y_pred_adv[exp_ind[pred_i]]], y2[exp_ind[pred_i],y_pred_adv[exp_ind[pred_i]]]),
                  fontsize=12)
plt.show()

pred_i = 5
label_gt = y_pred_pris[exp_ind[pred_i]]
confidence_pris = y1[exp_ind[pred_i],label_gt]
confidence_adv = y2[exp_ind[pred_i],label_gt]
confidence_pris_neighbor = exps['pris'][pred_i][3][:,label_gt]
confidence_adv_neighbor = exps['adv'][pred_i][3][:,label_gt]

print('mean of confidences in the vicinity of benign examples: %s'%np.mean(confidence_pris_neighbor))
print('mean of confidences in the vicinity of adversarial examples: %s'%np.mean(confidence_adv_neighbor))
print('std of confidences in the vicinity of benign examples: %s'%np.std(confidence_pris_neighbor))
print('std of confidences in the vicinity of adversarial examples: %s'%np.std(confidence_adv_neighbor))
plt.hist(confidence_pris_neighbor,alpha=0.5,label='pristine neighbor conf')
plt.hist(confidence_adv_neighbor,alpha=0.5,label='adversarial neighbor conf')
plt.axvline(x=confidence_pris,label='pristine conf',color='blue')
plt.axvline(x=confidence_adv,label='adversarial conf',color='orange')
plt.ylabel('Number of neighbors')
plt.xlabel('probability of predicting true label')
plt.legend()
plt.show()
