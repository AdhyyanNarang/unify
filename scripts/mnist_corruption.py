import tensorflow as tf
from skimage.filters import gaussian
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from cleverhans.attacks import FastGradientMethod

#gaussian blurring
#random epsilon perturbation
#random pixel blackout/whiteout on mnist

def gaussian_blurring(images, std_dev = 2):
	return np.array([gaussian(im, std_dev, preserve_range = True) for im in images])

def random_perturbation(images, eps = 0.5):
	return images + np.random.uniform(-eps, eps, images.shape)

def gaussian_perturbation(image, eps = 0.5):
        return image + np.random.normal(loc = 0, scale = eps, size = image.shape)

#p is probability that a pixel gets corrupted, p_b is probability that a corrupted pixel is blacked_out
def random_blackout_whiteout(images, p = 0.2, p_b = 0.5):
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
def corrupt_data(original_data, n, corrupt_func, seed = 0):
        np.random.seed(seed)
        data_copy = np.array(original_data)
        corrupt_indices = np.random.choice(list(range(len(data_copy))), n, replace = False)
        corrupted_pts = corrupt_func(data_copy[corrupt_indices])
        data_copy[corrupt_indices] = corrupted_pts
        return data_copy

