import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

import matplotlib.pyplot as plt

import gzip
from skimage.transform import resize
import numpy as np

def PreprocessImgs(imgs, target_size):
    new_imgs = np.zeros((len(imgs), target_size[0], target_size[1]))

    for i in range(len(imgs)):
        # FIXME anti_aliasing?
        # check what this resize is doing to the images
        new_imgs[i] = resize(imgs[i], target_size).astype('float32')
    return new_imgs


def LoadTrainData(target_shape):
    train_path = "./ndsb_dataset/images_train.npy.gz"
    labels_path = "./ndsb_dataset/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    train_idx = np.load("./ndsb_dataset/indices_train.npy")
    valid_idx = np.load("./ndsb_dataset/indices_valid.npy")


    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))
    new_shape = (len(imgs), target_shape[0], target_shape[1], target_shape[2])
    imgs = np.reshape(imgs, new_shape)

    X_train, y_train = imgs[train_idx], labels[train_idx]
    X_valid, y_valid = imgs[valid_idx], labels[valid_idx]

    return X_train, y_train, X_valid, y_valid


# load model from data
def LoadModel(in_shape, num_classes):
    # FIXME: 
    # - In the original network, the bias for convolutional layers is set to 1.0
    # - The original network uses leaky_relu and there is support for leaky_relu
    # in keras
    model = Sequential()

    a = 0.3

    # This looks like they call l1 in the code
    model.add(Conv2D(32, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they call l2 in the code
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they cal l3 in the code
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they call l4 in the code
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))

    # This looks like what they call l5
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model

num_classes= 121
img_shape = (95, 95, 1)

X_train, y_train, X_valid, y_valid = LoadTrainData(img_shape)

model = LoadModel(img_shape, num_classes)

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, -1, filter_indices=y_valid[0], seed_input=X_valid[0])
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')
plt.show()

