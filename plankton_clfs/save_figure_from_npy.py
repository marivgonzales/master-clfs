import matplotlib.pyplot as plt
import numpy as np
from math import floor
from skimage.transform import resize
import gzip

import os

#show the attention map of the test image
grads = np.load("./saliency_test_nounk.npy")

plt.imshow(grads, cmap='jet')
plt.show()


#looking for the test image
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
            current = resize(current, (target_size[0], minorside_new), mode='constant')#, anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,j,(k + minorside_pad)] = current[j,k]

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_size[1]), mode='constant')#, anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,(j + minorside_pad),k] = current [j,k]

        new_imgs[i] = new_imgs[i].astype('float32')
    return new_imgs

def LoadTrainData(target_shape):
    train_path = "./ndsb_dataset_nounk/images_train.npy.gz"
    labels_path = "./ndsb_dataset_nounk/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    #train_idx = np.load("./ndsb_dataset_nounk/indices_train.npy")
    valid_idx = np.load("./ndsb_dataset_nounk/indices_valid.npy")


    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))
    #new_shape = (len(imgs), target_shape[0], target_shape[1], target_shape[2])
    #imgs = np.reshape(imgs, new_shape)

    #X_train, y_train = imgs[train_idx], labels[train_idx]
    X_valid, y_valid = imgs[valid_idx], labels[valid_idx]

    return X_valid, y_valid

img_shape = (95, 95, 1)

X_valid, y_valid = LoadTrainData(img_shape)

print("Label of test image:", y_valid[0])

print(X_valid[0].shape)
plt.imshow(X_valid[0], cmap='gray')
plt.show()