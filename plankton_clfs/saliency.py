import keras
from keras.models import Sequential

from keras.models import model_from_json

from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

import numpy as np
from math import floor
from skimage.transform import resize
import gzip

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
    train_path = "./ndsb_dataset_nounk/images_train.npy.gz"
    labels_path = "./ndsb_dataset_nounk/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    #train_idx = np.load("./ndsb_dataset_nounk/indices_train.npy")
    valid_idx = np.load("./ndsb_dataset_nounk/indices_valid.npy")


    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))
    new_shape = (len(imgs), target_shape[0], target_shape[1], target_shape[2])
    imgs = np.reshape(imgs, new_shape)

    #X_train, y_train = imgs[train_idx], labels[train_idx]
    X_valid, y_valid = imgs[valid_idx], labels[valid_idx]

    return X_valid, y_valid

batch_size = 32
num_classes= 198

img_shape = (95, 95, 1)

X_valid, y_valid = LoadTrainData(img_shape)

# load json and create model
json_file = open('model_nounk.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_nounk.h5")
model.summary()

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, -1, filter_indices=y_valid[-1], seed_input=X_valid[-1])
grads1 = visualize_saliency(model, -1, filter_indices=y_valid[-2], seed_input=X_valid[-2])

# Save
np.save("./saliency_test_nounk1.npy", grads)
np.save("./saliency_test_nounk2.npy", grads1)
print("Images saved.")
