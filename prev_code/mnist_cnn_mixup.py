"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib

import requests
requests.packages.urllib3.disable_warnings()
import ssl
import pickle


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from attacks import fgm


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
n_sample_add = 1000

ind_all_3 = np.where(y_train == 0)[0]
ind_all_7 = np.where(y_train == 1)[0]

ind3_sub = np.random.choice(ind_all_3,n_sample_add)
ind7_sub = np.random.choice(ind_all_7,n_sample_add)
res = 0.1
ss = np.arange(0,1+res,res)
imgs = []
probs = []
for j in range(n_sample_add):
    ind3 = ind3_sub[j]
    ind7 = ind7_sub[j]
    X_sample_3 = X_train[ind3,:,:]
    X_sample_7 = X_train[ind7,:,:]
    for i in range(len(ss)):
        s = ss[i]
        img = X_sample_3 * (1-s) + X_sample_7 * s
        img = img.reshape((1,28,28,1))
        prob = s
        imgs.append(img)
        probs.append([1-s,s])

x_train_add = np.squeeze(np.array(imgs))
y_train_add = np.array(probs)
X_train = np.concatenate((X_train,x_train_add))

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
y_train = np.concatenate((y_train,y_train_add))
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

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=n_classes, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return  print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model_37_mixup', exist_ok=True)
        env.saver.save(sess, 'model_37_mixup/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv





print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
      name='mnist')

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

# print('\nGenerating adversarial data')
#
# X_adv = make_fgsm(sess, env, X_test, eps=0.02, epochs=12)
# # X_adv = make_deepfool(sess,env,X_test,epochs=3)
#
# print('\nEvaluating on adversarial data')

# evaluate(sess, env, X_adv, y_test)
#
# print('\nSaving adversarial data')




# print('\nRandomly sample adversarial data from each category')
#
# y1 = predict(sess, env, X_test)
# y2 = predict(sess, env, X_adv)

# with open('objs_37_deepfool.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([X_train,y_train,X_test,y_test,y1,X_adv,y2], f)
#
# z0 = np.argmax(y_test, axis=1)
# z1 = np.argmax(y1, axis=1)
# z2 = np.argmax(y2, axis=1)
#
# X_tmp = np.empty((n_classes, 28, 28))
# y_tmp = np.empty((n_classes, n_classes))
# for i in range(n_classes):
#     print('Target {0}'.format(i))
#     ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
#     cur = np.random.choice(ind)
#     X_tmp[i] = np.squeeze(X_adv[cur])
#     y_tmp[i] = y2[cur]
#
# print('\nPlotting results')
#
# fig = plt.figure(figsize=(n_classes, 1.2))
# gs = gridspec.GridSpec(1, n_classes, wspace=0.05, hspace=0.05)
#
# label = np.argmax(y_tmp, axis=1)
# proba = np.max(y_tmp, axis=1)
# for i in range(n_classes):
#     ax = fig.add_subplot(gs[0, i])
#     ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
#                   fontsize=12)
#
# print('\nSaving figure')
#
#
# gs.tight_layout(fig)
# os.makedirs('img', exist_ok=True)
# plt.savefig('img/deepfool_mnist_37.png')

