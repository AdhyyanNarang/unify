import numpy as np
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

# reliability diagram (consider testing dataset)
res = 0.1
centerpoints = np.arange(0,1+res,res)
prob_one_hot = nn_predict_proba_fn(X_test)
prob = prob_one_hot[:,1]
y_test_compact = y_test[:,1] # probability of class 0

# plt.hist(prob)
# plt.xlabel('predicted probabilities')

freq = np.empty(len(centerpoints)-1)
prob_avg = np.empty(len(centerpoints)-1)
for i in range(len(centerpoints)-1):
    bin_l = centerpoints[i]
    bin_r = centerpoints[i+1]
    print(bin_l)
    print(bin_r)
    if i == len(centerpoints)-1:
        ind = np.where(np.logical_and(prob >= bin_l,prob <= bin_r)== True)[0]
    else:
        ind = np.where(np.logical_and(prob >= bin_l,prob <= bin_r)== True)[0]

    # print(ind)
    label = y_test_compact[ind]
    print(label)

    if len(ind) == 0:
        prob_avg[i] = np.nan
        freq[i] = np.nan
    else:
        prob_avg[i] = np.mean(prob[ind])
        freq[i] = np.sum(label == 1) / len(ind)

# plt.plot(freq,freq)
plt.plot((0,1),(0,1),linewidth=1)
plt.scatter(prob_avg,freq)
plt.xlabel('forecast probability of label=1')
plt.ylabel('observed relative frequency of label=1')
plt.title('reliability diagram after data augmentation')