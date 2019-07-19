# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import regularizers
from keras import models

# Helper libraries
import numpy as np
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from openTSNE import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



"""
Helpful functions for MNIST
"""
def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )

"""
Model creation and evaluation functions
"""

def create_model_one(input_shape, num_classes, logits = False, input_ph = None):
    model = Sequential()
    layers = [Conv2D(32, (5,5), input_shape = input_shape),
            Activation('relu'),
            Conv2D(32, kernel_size = (3,3)),
            Activation('relu'),
            MaxPooling2D(pool_size = (2,2)),
            #Dropout(0.25),
            Conv2D(64, (3,3)),
            Activation('relu'),
            Conv2D(64, (3,3)),
            Activation('relu'),
            MaxPooling2D(pool_size = (2,2)),
            #Dropout(0.25),
            Flatten(),
            Dense(1024),
            Activation('relu'),
            Dense(1024),
            Activation('relu'),
            Dense(num_classes)]
    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

def create_fully_connected(input_shape, num_hidden, num_classes, reg = 0.0, logits = False, input_ph = None):
    model = Sequential()
    layers = [Dense(units = 32, input_shape= input_shape, activation = 'sigmoid', name = 'first', kernel_regularizer = regularizers.l2(reg))
    ]

    for i in range(num_hidden):
        layer_name = 'hidden_' + str(i) 
        layers.append(Dense(units = 32, name = layer_name, activation = 'sigmoid',kernel_regularizer = regularizers.l2(reg)))

    layers.append(Dense(units = num_classes, name = 'last', activation = 'softmax', kernel_regularizer = regularizers.l2(reg)))

    for layer in layers:
        model.add(layer)

    return model

def create_fully_connected_k_onwards(input_shape, new_input_shape, k, num_hidden, num_classes, reg = 0.0, logits = False, input_ph = None):
    model = Sequential()
    
    layers = [Dense(units = 32, input_shape= input_shape, activation = 'sigmoid', name = 'first', kernel_regularizer = regularizers.l2(reg))
    ]

    for i in range(num_hidden):
        layer_name = 'hidden_' + str(i) 
        layers.append(Dense(units = 32, activation = 'sigmoid', name = layer_name, kernel_regularizer = regularizers.l2(reg)))

    layers.append(Dense(units = num_classes, name = 'last', activation = 'softmax', kernel_regularizer = regularizers.l2(reg)))

    layers = layers[k:]
    layers[0] = Dense(units = 32, input_shape = new_input_shape, activation = 'sigmoid', name = layers[0].name, kernel_regularizer = regularizers.l2(reg))

    for layer in layers:
        model.add(layer)

    return model

def create_fully_connected_SVM_multiclass(input_shape, num_hidden, num_classes, reg = 0.0, logits = False, input_ph = None):
    model = Sequential()
    layers = [Dense(units = 32, input_shape= input_shape, activation = 'sigmoid', name = 'first')
    ]

    for i in range(num_hidden):
        layer_name = 'hidden_' + str(i) 
        layers.append(Dense(units = 32, name = layer_name, activation = 'sigmoid'))

    layers.append(Dense(units = num_classes, name = 'last', activation = 'linear', kernel_regularizer = regularizers.l2(reg)))

    for layer in layers:
        model.add(layer)

    return model

def create_fully_connected_SVM_binary(input_shape, num_hidden, reg = 0.0, logits = False, input_ph = None):
    model = Sequential()
    layers = [Dense(units = 32, input_shape= input_shape, activation = 'sigmoid', name = 'first')
    ]

    for i in range(num_hidden):
        layer_name = 'hidden_' + str(i) 
        layers.append(Dense(units = 32, name = layer_name, activation = 'sigmoid'))

    layers.append(Dense(units = 1, name = 'last', activation = 'linear', kernel_regularizer = regularizers.l2(reg)))

    for layer in layers:
        model.add(layer)

    return model


#Uses adversarial examples from cleverhans
def train_and_test(model_input, X_input, y_input, epochs = 2, eps_test = 0.2):
    model_input.fit(X_input, y_input, epochs = epochs)
    print('finished training')
    test_loss, test_acc = model_input.evaluate(test_images, test_labels)
    #Evaluation of normal model on adversarial test examples
    print('Test accuracy:', test_acc)
    #Evaluation of adversarially trained model
    acc, adv_x_np, preds = adv_evaluate(model_input, 0.2)
    print('Test accuracy on adversarial examples:', acc)



"""
Adversarial Functions
"""

def adv_evaluate(model_input, epsilon_input):
    wrap = KerasModelWrapper(model_input)
    fgsm = FastGradientMethod(wrap)
    adv_x = fgsm.generate_np(test_images, eps = epsilon_input, clip_min = -2, clip_max = 2)
    preds_adv = model_input.predict(adv_x)
    eval_par = {'batch_size': 60000}
    test_loss, acc = model_input.evaluate(adv_x, test_labels)
    return acc, adv_x, preds_adv

#Generate adversarial examples
def adv_generate(session, model_input, epsilon_input, image_set):
    fgsm = FastGradientMethod(model_input, sess=session)
    adv_x = fgsm.generate_np(image_set, eps = epsilon_input, clip_min = -2, clip_max = 2)
    return adv_x


"""
Norm functions
"""
#Relevant norm for adversarial examples no. 1
def sum_entries_norm(X):
    return sum(sum(abs(X)))

"""
Distance functions
"""

#Distance related functions

def dist_calculator(reg_image, adv_image, activation_model):
    reg_image = reg_image.reshape(1, 784)
    adv_image = adv_image.reshape(1, 784)
    activation_reg = np.array(activation_model.predict(reg_image)).squeeze()
    activation_adv = np.array(activation_model.predict(adv_image)).squeeze()
    difference = activation_reg - activation_adv
    L = len(difference)
    if L == 1:
        return np.linalg.norm(activation_reg - activation_adv)
    else:
        distances = []
        for i in range(L):
            distances.append(np.linalg.norm(difference[i]))
        return distances

def dist_average(reg_set, adv_set, activation_model):
    set_card = len(reg_set)
    distances = []
    for i in range(set_card):
        dist = dist_calculator(reg_set[i], adv_set[i], activation_model)
        distances.append(dist)
    #print(np.array(distances).shape)
    return np.average(distances, axis = 0)

def dist_split(x_test_flat, x_test_adv_flat, y_test, model, activation_model):
    pred = model.predict(x_test_adv_flat)
    incorrect_indices = [i for i,v in enumerate(pred) if np.argmax(pred[i]) != np.argmax(y_test[i])]
    correct_indices = [i for i,v in enumerate(pred) if np.argmax(pred[i]) == np.argmax(y_test[i])]
    
    overall = dist_average(x_test_flat, x_test_adv_flat, activation_model)
    correct_dist = dist_average(x_test_flat[correct_indices], x_test_adv_flat[correct_indices], activation_model)
    incorrect_dist = dist_average(x_test_flat[incorrect_indices], x_test_adv_flat[incorrect_indices], activation_model)
    
    return overall, correct_dist, incorrect_dist

"""
Visualization functions
"""

#TODO: Fix undefined variable errors in the two functions below. Look at cv2 tutorial code.
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

def generate_data_to_plot(sess, trained_model, x_train, y_train, eps, corruption_type = 'adv', tsne_flag = True):
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
    L = len(trained_model.layers)
    layer_output = [trained_model.layers[i].output for i in range(L - 1)]
    activation_model = models.Model(inputs=trained_model.input, outputs=layer_output)
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
    plt.savefig(save_path)
