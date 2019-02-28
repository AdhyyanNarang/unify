import tensorflow as tf
from skimage.filters import gaussian
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from cleverhans.attacks import FastGradientMethod

#gaussian blurring
#random epsilon perturbation
#random pixel blackout/whiteout on mnist

def gaussian_blurring(images, std_dev):
	return np.array([gaussian(im, std_dev) for im in images])

def random_perturbation(images, eps):
	return images + np.random.uniform(-eps, eps, images.shape)

#p is probability that a pixel gets corrupted, p_b is probability that a corrupted pixel is blacked_out
def random_blackout_whiteout(images, p, p_b):
	#0 corresponds to blackout, 2 corresponds to whiteout, 1 means do nothing
	mask = np.random.choice([0,1,2], images.shape, replace = True, p = [p * p_b, 1 - p, p * (1 - p_b)])

	blackout_mask = np.array(mask)
	blackout_mask[blackout_mask == 2] = 0
	blackout = images * blackout_mask

	whiteout_mask = np.array(mask)
	whiteout_mask[whiteout_mask == 1] = 0
	whiteout_mask[whiteout_mask == 2] = 1
	whiteout = np.maximum(blackout, whiteout_mask)

	return whiteout

#using FGSM method
def adversarial_examples(images, model, sess):
	fgsm = FastGradientMethod(model, sess = sess)
	adversarial_images = fgsm.generate_np(images, clip_min = -1, clip_max = 1)
	return adversarial_images

#corrupt n points
def corrupt_data(original_data, corrupted_data, n, seed = 0):
	np.random.seed(seed)
	shuffle_indices = list(range(len(original_data)))
	np.random.shuffle(shuffle_indices)
	original_data = original_data[shuffle_indices]
	corrupted_data = corrupted_data[shuffle_indices]
	return np.vstack((corrupted_data[:n], original_data[n:]))





mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

temp_set = x_train[:]
blurred = gaussian_blurring(temp_set, 2)
perturbed = random_perturbation(temp_set, 0.1)
black_white = random_blackout_whiteout(temp_set, 0.2, 0.5)

corrupted_set = corrupt_data(temp_set, blurred, 20)
print(corrupted_set.shape)

plt.imshow(temp_set[0], cmap = "gray")
plt.show()

plt.imshow(blurred[0], cmap = "gray")
plt.show()

plt.imshow(perturbed[0], cmap = "gray")
plt.show()

plt.imshow(black_white[0], cmap = "gray")
plt.show()

plt.imshow(corrupted_set[19], cmap = "gray")
plt.show()

plt.imshow(corrupted_set[20], cmap = "gray")
plt.show()

