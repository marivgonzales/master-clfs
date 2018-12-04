import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

import numpy as np
import gzip
import math

from sklearn import preprocessing

import os

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

def LoadModel(in_shape, num_classes):
    # FIXME: 
    # - In the original network, the bias for convolutional layers is set to 1.0
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

def step_decay(epoch):
    if epoch < 100:
        return 0.003
    elif epoch < 200:
        return 0.0015
    elif epoch < 350:
        return 0.0003
    elif epoch < 500:
        return 0.0001
    else:
        return 0.00003


batch_size = 32
num_classes= 121

img_shape = (95, 95, 1)

X_train, y_train, X_valid, y_valid = LoadTrainData()
X_train = X_train.astype("float32")
X_valid = X_valid.astype("float32")
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)


datagen = ImageDataGenerator(
                             rotation_range=60,
                             featurewise_center=True,
                             featurewise_std_normalization=True,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.2,
                             shear_range=10,
                             horizontal_flip=True,
                             vertical_flip=True)

datagen.fit(X_train)
X_valid = datagen.standardize(X_valid)

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

model = LoadModel(img_shape, num_classes)
model.summary()

opt = keras.optimizers.SGD(lr=0.0, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model_path = "best_model_079.hdf5" 

checkpoint = ModelCheckpoint(model_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')

lrate = LearningRateScheduler(step_decay)

model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_data=(X_valid, y_valid),
                    validation_steps=len(X_valid) // batch_size,
                    epochs=800,
                    callbacks=[lrate, checkpoint])

model_json = model.to_json()
with open("model_079.json","w") as json_file:
    json_file.write(model_json)
print('Saved trained model at best_model_079.hdf5 and model_079.json')

predictions = model.predict(X_valid, batch_size=batch_size)
predicted_labels = np.argmax(predictions, axis=1)
np.save("./ndsb_dataset/complete_predictions_valid.npy", predictions)
np.save("./ndsb_dataset/predicted_labels_valid.npy", predicted_labels)
np.save("./ndsb_dataset_tax/real_labels_valid.npy", Y_valid)

predictions = model.predict(X_train, batch_size=batch_size)
predicted_labels = np.argmax(predictions, axis=1)
np.save("./ndsb_dataset/complete_predictions_train.npy", predictions)
np.save("./ndsb_dataset/predicted_labels_train.npy", predicted_labels)
np.save("./ndsb_dataset_tax/real_labels_train.npy", Y_train)
print("Predictions saved")

print("Predictions saved")



