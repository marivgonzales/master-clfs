from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import gzip
import math
from math import floor
from skimage.transform import resize
import cv2
import os

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
            current = resize(current, (target_size[0], minorside_new), mode='constant', anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,j,(k + minorside_pad)] = current[j,k]

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_size[1]), mode='constant', anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,(j + minorside_pad),k] = current [j,k]

        new_imgs[i] = new_imgs[i].astype('float32')

    
    return new_imgs

def LoadTrainData(target_shape):
    train_path = "./laps_nobg_100/images_train.npy.gz"
    labels_path = "./laps_nobg_100/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))

    focused_images = np.zeros((0,target_shape[0], target_shape[1]))
    focused_labels = np.zeros((0))

    for i in range(len(imgs)):
        focus_meas = TENG(imgs[i])
        focus_measure = np.log(focus_meas)
        if focus_measure > -2.3:
            img = np.reshape(imgs[i], (1, target_shape[0], target_shape[1]))
            focused_images = np.append(focused_images, img, axis=0)
            print("Adicionando imagem,")
            focused_labels = np.append(focused_labels, labels[i])
            print("Adicionando label", labels[i])

    new_shape = (len(focused_images), target_shape[0], target_shape[1], target_shape[2])
    focused_images = np.reshape(focused_images, new_shape)

    classes_real, counts_reall = np.unique(focused_labels, return_counts=True)
    print("Classes: ", classes_real)
    print("Counts: ", counts_reall)

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.1)
    train_idx, valid_idx = next(split.split(np.zeros(len(focused_labels)), focused_labels))

    X_train, y_train = focused_images[train_idx], focused_labels[train_idx]
    X_valid, y_valid = focused_images[valid_idx], focused_labels[valid_idx]

    return X_train, y_train, X_valid, y_valid


img_shape = (95, 95, 1)

X_train, y_train, X_valid, y_valid = LoadTrainData(img_shape)

print(np.shape(y_train))
print(np.shape(y_valid))


np.save("./laps_images_train_ffilter_2-3.npy", X_train)
np.save("./laps_real_labels_train_ffilter_2-3.npy", y_train)

np.save("./laps_images_valid_ffilter_2-3.npy", X_valid)
np.save("./laps_real_labels_valid_ffilter_2-3.npy", y_valid)

print("Predictions saved")



