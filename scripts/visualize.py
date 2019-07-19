import sys
sys.path.insert(0, '../')
import tensorflow as tf
import keras
from keras import models
from keras import regularizers
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.manifold import TSNE
from openTSNE import TSNE
import pandas as pd
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from mnist_corruption import gaussian_blurring, corrupt_data, random_perturbation, random_blackout_whiteout
from adv_util import create_fully_connected, create_fully_connected_k_onwards
from sklearn.decomposition import PCA
import cv2

#Config
trans_x = 5
trans_y = 10
rotate_angle = 45


"""
Helper Functions
"""
def rotate(image_set, angle):
    destination_set = image_set.copy() 
    for (idx, img) in enumerate(image_set):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        destination_set[idx] = dst
    return destination_set

def translate(image_set, x, y):
    destination_set = image_set.copy() 
    for (idx, img) in enumerate(image_set):
        M = np.float32([[1,0,x],[0,1,y]])
        dst = cv2.warpAffine(img,M,(cols,rows))
        destination_set[idx] = dst
    return destination_set

def adv_generate(session, model_input, epsilon_input, image_set):
    fgsm = FastGradientMethod(model_input, sess=session)
    adv_x = fgsm.generate_np(image_set, eps = epsilon_input, clip_min = -2, clip_max = 2)
    return adv_x

#TODO: Make function more general by
#taking corruption_type and tsne_flag into account
def generate_data_to_plot(trained_model, x_train, y_train, eps, corruption_type = 'adv', tsne_flag = True):
    #Generate corrupted input images
    x_train_adv = None 
    if corruption_type == 'adv':
        x_train_adv = adv_generate(sess, KerasModelWrapper(trained_model), eps, x_train)
        trained_model.evaluate(x_train_adv, y_train)
    elif corruption_type == 'rotate':
        x_train_adv = rotate(x_train, rotate_angle)
    elif corruption_type == 'translate':
        x_train_adv = translate(x_train, trans_x, trans_y) 
    else:
        x_train_adv = x_train

    #Create array of activations at each layer
    L = len(model.layers)
    layer_output = [model.layers[i].output for i in range(L - 1)]
    activation_model = models.Model(inputs=model.input, outputs=layer_output)
    activations = activation_model.predict(x_train_adv)

    tsne_activations = []
    tsne = None
    #Reduce dimensionality of each (Either TSNE or PCA)
    if tsne_flag:
        #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne = TSNE()
    else:
        tsne = PCA(n_components = 2)

    for a_l in activations:
        #tsne_results = tsne.fit_transform(a_l)
        tsne_results = tsne.fit(a_l)
        tsne_activations.append(tsne_results)

    return tsne_activations


def plot_data(tsne_results, y, save_path):
    df = pd.DataFrame(y, columns = ['y'])
    columns = ['tsne-one', 'tsne-two']
    df[columns[0]] = tsne_results[:, 0]
    df[columns[1]] = tsne_results[:, 1]
    df['y'] = y

    plt.figure(figsize=(16,10))
    sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=0.3,
    )
    plt.savefig(savepath)

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )

if __name__ == '__main__':
    sess = tf.keras.backend.get_session()
    keras.backend.set_session(sess)
    #Dataset setup stuff
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    num_classes = 10
    y_train_old = y_train
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train_flat, input_shape = flatten_mnist(x_train)
    x_test_flat, _ = flatten_mnist(x_test)

    #Train the model
    model = create_fully_connected(input_shape = input_shape, num_classes = num_classes, num_hidden = 4, reg = 0)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    sess.run(tf.global_variables_initializer())
    #model.fit(x_train_flat, y_train, epochs = 15)
    #model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    print(model.evaluate(x_test_flat, y_test))
    x_test_flat_adv = adv_generate(sess, KerasModelWrapper(model), 0.10 , x_test_flat)
    print(model.evaluate(x_test_flat_adv, y_test))

    #Obtain TSNE activations
    for eps in [0.10, 0.15]:
        save_directory = "img/tsne/adversarial/" + str(eps)
        tsne_activations = generate_data_to_plot(model, x_train_flat, y_train, eps = eps, tsne_flag = True)
        #np.save("tsne_ac_05", tsne_activations)
        #tsne_activations = np.load("tsne_ac.npy")

        #Plot the TSNE activations and save the images
        for (idx, tsne_ac) in enumerate(tsne_activations):
            savepath = save_directory + "/layer_" + str(idx)
            plot_data(tsne_ac, y_train_old, savepath)
