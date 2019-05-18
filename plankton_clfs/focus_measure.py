import cv2
import numpy as np
#from utils import Data
from math import floor
from skimage.transform import resize
import gzip
import scipy

import matplotlib.pyplot as plt


def LAPV(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return np.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2


def LAPM(img):
    """Implements the Modified Laplacian (LAP2) focus measure
    operator. Measures the amount of edges present in the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    kernel = np.array([-1, 2, -1])
    laplacianX = np.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = np.abs(cv2.filter2D(img, -1, kernel.T))
    return np.mean(laplacianX + laplacianY)


def TENG(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return np.mean(gaussianX * gaussianX +
                      gaussianY * gaussianY)

"""
def MLOG(img):
    Implements the MLOG focus measure algorithm.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))
"""

def GBLVAR(img):
    return np.var(scipy.ndimage.uniform_filter(img))

def TOTVAR(img):
    return np.sum(scipy.ndimage.uniform_filter(img))

# Loading data
"""
train_path = "./laps_nobg_100/images_train.npy.gz"
labels_path = "./laps_nobg_100/labels_train.npy.gz"
train_idx = np.load("./laps_nobg_100/indices_train.npy")
valid_idx = np.load("./laps_nobg_100/indices_valid.npy")

target_shape = (95, 95, 1)
data = Data()
img_objects = data.load(train_path, labels_path, target_shape)

data.set_train_val_indices(train_idx, valid_idx)
#X_train, y_train = data.create_train_set()
X_valid, y_valid = data.create_valid_set()
#train_objects = img_objects[data.get_train_indices()]
valid_objects = img_objects[data.get_valid_indices()]
"""

def PreprocessImgs(imgs, target_size):
    new_imgs = np.ones((len(imgs), target_size[0], target_size[1]))

    for i in range(len(imgs)):
        current = imgs[i]
        majorside = np.amax(current.shape)
        majorside_idx = np.argmax(current.shape)
        minorside = np.amin(current.shape)

        factor = target_size[0]/majorside
        minorside_new = floor(minorside*factor)
        minorside_pad = floor((target_size[0] - minorside_new)/2)
        
        if majorside_idx == 0:
            current = resize(current, (target_size[0], minorside_new), mode='constant')
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,j,(k + minorside_pad)] = current[j,k]

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_size[1]), mode='constant')
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,(j + minorside_pad),k] = current [j,k]

        new_imgs[i] = new_imgs[i].astype('float32')
    return new_imgs

def LoadTrainData(target_shape):
    train_path = "./laps_nobg_100/images_train.npy.gz"
    labels_path = "./laps_nobg_100/labels_train.npy.gz"

    #train_path = "./ndsb_dataset_nounk/images_train.npy.gz"
    #labels_path = "./ndsb_dataset_nounk/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    train_idx = np.load("./laps_nobg_100/indices_train.npy")
    valid_idx = np.load("./laps_nobg_100/indices_valid.npy")

    #train_idx = np.load("./ndsb_dataset_nounk/indices_train.npy")
    #valid_idx = np.load("./ndsb_dataset_nounk/indices_valid.npy")


    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)

    """
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))
    new_shape = (len(imgs), target_shape[0], target_shape[1], target_shape[2])
    imgs = np.reshape(imgs, new_shape)
    """

    X_train, y_train = imgs[train_idx], labels[train_idx]
    X_valid, y_valid = imgs[valid_idx], labels[valid_idx]

    X_train = PreprocessImgs(X_train, (target_shape[0], target_shape[1]))
    new_shape = (len(X_train), target_shape[0], target_shape[1], target_shape[2])
    X_train = np.reshape(X_train, new_shape)


    return X_train


img_shape = (95, 95, 1)

X_valid = LoadTrainData(img_shape)


f = np.zeros((len(X_valid), 5))

for i in range(len(X_valid)):
    f[i,0] = LAPV(X_valid[i])
    f[i,1] = LAPM(X_valid[i])
    f[i,2] = TENG(X_valid[i])
    f[i,3] = GBLVAR(X_valid[i])
    f[i,4] = TOTVAR(X_valid[i])

print(f)
np.save("./focus_train_laps.npy", f)
