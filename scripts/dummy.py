import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import cv2
import keras


(x_train,y_train), _ = keras.datasets.mnist.load_data()
img = x_train[0]
rows,cols = img.shape

"""
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Translation
M = np.float32([[1,0,10],[0,1,5]])
dst = cv2.warpAffine(img,M,(cols,rows))
"""
#Rotation
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


